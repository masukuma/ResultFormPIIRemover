//! Files on disk: where they are kept (by id, under their account, never
//! by the client's name), what kind they are (by their first bytes, not
//! their name or declared type), and the checks a zip must pass.

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

/// The kinds of file `rfpii` reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Pdf,
    Jpeg,
    Png,
    Tiff,
    Zip,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Pdf => "pdf",
            Kind::Jpeg => "jpeg",
            Kind::Png => "png",
            Kind::Tiff => "tiff",
            Kind::Zip => "zip",
        }
    }

    pub fn parse(s: &str) -> Option<Kind> {
        [Kind::Pdf, Kind::Jpeg, Kind::Png, Kind::Tiff, Kind::Zip].into_iter().find(|k| k.name() == s)
    }

    /// The extension a file of this kind gets, which `rfpii` reads it by.
    pub fn ext(self) -> &'static str {
        match self {
            Kind::Pdf => "pdf",
            Kind::Jpeg => "jpg",
            Kind::Png => "png",
            Kind::Tiff => "tif",
            Kind::Zip => "zip",
        }
    }

    /// The extensions a client's name may keep for this kind.
    fn exts(self) -> &'static [&'static str] {
        match self {
            Kind::Pdf => &["pdf"],
            Kind::Jpeg => &["jpg", "jpeg"],
            Kind::Png => &["png"],
            Kind::Tiff => &["tif", "tiff"],
            Kind::Zip => &["zip"],
        }
    }
}

/// Bytes read from the start of a file to tell its kind.
pub const HEAD: usize = 1024;

/// The kind of a file from its first bytes. A PDF's header may come
/// anywhere in its first 1024 bytes (ISO 32000-1 §7.5.2, as readers allow).
pub fn sniff(head: &[u8]) -> Option<Kind> {
    if head.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(Kind::Jpeg)
    } else if head.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        Some(Kind::Png)
    } else if head.starts_with(b"II*\0") || head.starts_with(b"MM\0*") {
        Some(Kind::Tiff)
    } else if head.starts_with(b"PK\x03\x04") {
        Some(Kind::Zip)
    } else if head.windows(5).any(|w| w == b"%PDF-") {
        Some(Kind::Pdf)
    } else {
        None
    }
}

/// Where an uploaded file is kept.
pub fn file_path(data_dir: &Path, account: &str, file_id: &str) -> PathBuf {
    data_dir.join("files").join(account).join(file_id)
}

/// A job's folder.
pub fn job_dir(data_dir: &Path, account: &str, job_id: &str) -> PathBuf {
    data_dir.join("jobs").join(account).join(job_id)
}

/// Uploads in progress, on the same file system as `files/`.
pub fn tmp_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("tmp")
}

/// The client's name for a file, if it is safe to name the file by in a
/// job's input folder: letters, digits, `.`, `_` and `-` only, at most 128
/// characters, not starting with a dot, and ending in an extension of its
/// kind. Otherwise None, and the file is named by its id.
pub fn safe_name(client: &str, kind: Kind) -> Option<String> {
    let base = client.rsplit(['/', '\\']).next().unwrap_or("");
    let ok_chars = base.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    let ext = base.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    let ok_ext = ext.as_deref().is_some_and(|e| kind.exts().contains(&e));
    (ok_chars && ok_ext && !base.starts_with('.') && base.len() <= 128 && !base.contains("..")).then(|| base.to_string())
}

/// An upload being written to disk: its size is counted against a limit,
/// it is hashed, and its first bytes kept to tell its kind.
pub struct Sink {
    pub path: PathBuf,
    file: tokio::fs::File,
    hasher: Sha256,
    pub bytes: u64,
    pub head: Vec<u8>,
    limit: u64,
}

/// What a finished upload is.
pub struct Received {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    pub head: Vec<u8>,
}

/// The upload went over the limit.
#[derive(Debug)]
pub struct TooLarge;

impl Sink {
    pub async fn create(data_dir: &Path, limit: u64) -> std::io::Result<Sink> {
        let dir = tmp_dir(data_dir);
        tokio::fs::create_dir_all(&dir).await?;
        let path = dir.join(crate::ids::new_id("upload"));
        let file = tokio::fs::File::create(&path).await?;
        Ok(Sink { path, file, hasher: Sha256::new(), bytes: 0, head: Vec::new(), limit })
    }

    /// Write a chunk; Err(TooLarge) once the limit is passed.
    pub async fn write(&mut self, chunk: &[u8]) -> Result<std::io::Result<()>, TooLarge> {
        self.bytes += chunk.len() as u64;
        if self.bytes > self.limit {
            return Err(TooLarge);
        }
        if self.head.len() < HEAD {
            let take = (HEAD - self.head.len()).min(chunk.len());
            self.head.extend_from_slice(&chunk[..take]);
        }
        self.hasher.update(chunk);
        Ok(self.file.write_all(chunk).await)
    }

    pub async fn finish(mut self) -> std::io::Result<Received> {
        self.file.flush().await?;
        self.file.sync_all().await?;
        Ok(Received { path: self.path, bytes: self.bytes, sha256: crate::ids::hex(&self.hasher.finalize()), head: self.head })
    }

    /// Remove what was written.
    pub async fn discard(self) {
        drop(self.file);
        let _ = tokio::fs::remove_file(&self.path).await;
    }
}

/// What a zip holds, once it passed the checks.
#[derive(Debug, PartialEq)]
pub struct ZipSummary {
    pub entries: u64,
    pub uncompressed_bytes: u64,
}

/// Check a zip before it is accepted: at most `max_entries` entries; no
/// entry named with an absolute path, a `..` component, a backslash or a
/// NUL; none encrypted or a symbolic link; and no more than `max_bytes`
/// once expanded, counted by expanding it (sizes a zip states can lie).
pub fn check_zip(path: &Path, max_entries: u64, max_bytes: u64) -> Result<ZipSummary, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("can't open the zip: {e}"))?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| format!("not a readable zip: {e}"))?;
    if z.len() as u64 > max_entries {
        return Err(format!("{} entries; at most {max_entries} are accepted", z.len()));
    }
    let mut total: u64 = 0;
    for i in 0..z.len() {
        let mut e = z.by_index(i).map_err(|e| format!("entry {i} can't be read: {e}"))?;
        let name = e.name().map(|n| n.to_string()).map_err(|err| format!("entry {i} has an unreadable name: {err}"))?;
        unsafe_entry_name(&name).map_or(Ok(()), Err)?;
        if e.encrypted() {
            return Err(format!("entry {name:?} is encrypted"));
        }
        if e.is_symlink() {
            return Err(format!("entry {name:?} is a symbolic link"));
        }
        let left = max_bytes - total;
        let n = std::io::copy(&mut (&mut e).take(left + 1), &mut std::io::sink()).map_err(|err| format!("entry {name:?} can't be expanded: {err}"))?;
        total += n;
        if total > max_bytes {
            return Err(format!("expands to more than {max_bytes} bytes"));
        }
    }
    Ok(ZipSummary { entries: z.len() as u64, uncompressed_bytes: total })
}

/// Why an entry name is unsafe to expand, if it is.
fn unsafe_entry_name(name: &str) -> Option<String> {
    let bad = name.starts_with('/') || name.contains('\\') || name.contains('\0') || name.split('/').any(|c| c == "..") || name.chars().nth(1) == Some(':');
    bad.then(|| format!("entry {name:?} would be written outside the folder it is expanded in"))
}

/// Delete a file or folder if it is there; whether it was there. Another
/// process deleting it at the same time is not an error.
pub fn remove(path: &Path) -> std::io::Result<bool> {
    let gone = |r: std::io::Result<()>| match r {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(true),
    };
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.is_dir() => gone(std::fs::remove_dir_all(path)),
        Ok(_) => gone(std::fs::remove_file(path)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn kinds_are_told_by_their_first_bytes() {
        assert_eq!(sniff(b"%PDF-1.7\n"), Some(Kind::Pdf));
        assert_eq!(sniff(b"\n\n%PDF-1.4"), Some(Kind::Pdf));
        assert_eq!(sniff(&[0xFF, 0xD8, 0xFF, 0xE0]), Some(Kind::Jpeg));
        assert_eq!(sniff(b"\x89PNG\r\n\x1a\n...."), Some(Kind::Png));
        assert_eq!(sniff(b"II*\0...."), Some(Kind::Tiff));
        assert_eq!(sniff(b"MM\0*...."), Some(Kind::Tiff));
        assert_eq!(sniff(b"PK\x03\x04...."), Some(Kind::Zip));
        assert_eq!(sniff(b"PK\x05\x06"), None); // an empty zip
        assert_eq!(sniff(b"<html>"), None);
    }

    #[test]
    fn a_client_name_is_kept_only_if_safe_and_of_its_kind() {
        assert_eq!(
            safe_name("1_34_A_007029014000102_F230450M00203609_20220810_132155.pdf", Kind::Pdf).as_deref(),
            Some("1_34_A_007029014000102_F230450M00203609_20220810_132155.pdf")
        );
        assert_eq!(safe_name("C:\\scans\\form.JPG", Kind::Jpeg).as_deref(), Some("form.JPG"));
        assert_eq!(safe_name("form.pdf", Kind::Jpeg), None);
        assert_eq!(safe_name(".hidden.pdf", Kind::Pdf), None);
        assert_eq!(safe_name("a b.pdf", Kind::Pdf), None);
        assert_eq!(safe_name("x..pdf", Kind::Pdf), None);
        assert_eq!(safe_name("noext", Kind::Pdf), None);
    }

    fn zip_of(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for (name, data) in entries {
            w.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            w.write_all(data).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    #[test]
    fn a_zip_is_checked_for_entries_size_and_names() {
        let dir = std::env::temp_dir().join(format!("rfpii-api-zip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("z.zip");
        std::fs::write(&p, zip_of(&[("a/x.pdf", b"%PDF-1.4"), ("b.jpg", &[0u8; 1000])])).unwrap();
        assert_eq!(check_zip(&p, 10, 10_000), Ok(ZipSummary { entries: 2, uncompressed_bytes: 1008 }));
        assert!(check_zip(&p, 1, 10_000).unwrap_err().contains("2 entries"));
        assert!(check_zip(&p, 10, 1000).unwrap_err().contains("expands to more"));
        for bad in ["../x.pdf", "/etc/x.pdf", "a/../../x.pdf", "a\\..\\x.pdf"] {
            std::fs::write(&p, zip_of(&[(bad, b"%PDF")])).unwrap();
            assert!(check_zip(&p, 10, 10_000).unwrap_err().contains("outside"), "{bad}");
        }
        std::fs::write(&p, b"PK\x03\x04 not really").unwrap();
        assert!(check_zip(&p, 10, 10_000).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
