//! Digital signatures on the redacted PDFs, as PDF readers check them.
//!
//! The signature is an incremental update appended to the PDF the pipeline
//! wrote, so those bytes stay as they are. The update adds an invisible
//! signature field whose value is a detached PKCS#7 signature
//! (`adbe.pkcs7.detached`) over the whole file except the signature itself.
//! Adobe Acrobat/Reader, pdfsig (poppler), pyHanko and `rfpii verify` check it.
//!
//! A certificate from a CA that readers already trust is best. Without one,
//! `rfpii certgen` creates a self-signed certificate. Readers then show the
//! signature as intact but from an unknown signer, so the certificate's
//! SHA-256 fingerprint must be published where people can compare it.

use anyhow::{anyhow, bail, Context, Result};
use lopdf::{Dictionary, Document, Object};
use openssl::asn1::{Asn1Integer, Asn1Time};
use openssl::bn::{BigNum, MsbOption};
use openssl::hash::MessageDigest;
use openssl::nid::Nid;
use openssl::pkcs12::Pkcs12;
use openssl::pkcs7::{Pkcs7, Pkcs7Flags};
use openssl::pkey::{PKey, Private};
use openssl::rsa::Rsa;
use openssl::stack::Stack;
use openssl::x509::extension::{BasicConstraints, ExtendedKeyUsage, KeyUsage, SubjectKeyIdentifier};
use openssl::x509::store::{X509Store, X509StoreBuilder};
use openssl::x509::{X509NameBuilder, X509NameRef, X509Ref, X509StoreContext, X509};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Shown by PDF readers as the reason for signing.
pub const REASON: &str = "Personal data removed; all other content unchanged";
const FIELD: &str = "rfpii";
/// id-kp-documentSigning (RFC 9336)
const DOCUMENT_SIGNING: &str = "1.3.6.1.5.5.7.3.36";

/// A certificate and its private key.
pub struct Signer {
    key: PKey<Private>,
    cert: X509,
    chain: Stack<X509>,
}

/// Where `rfpii certgen` puts the certificate by default:
/// $XDG_CONFIG_HOME/rfpii/signer.p12 (~/.config/rfpii/signer.p12).
pub fn default_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("rfpii").join("signer.p12")
}

/// The public certificate written beside a PKCS#12 file.
pub fn certificate_path(p12: &Path) -> PathBuf {
    p12.with_extension("crt")
}

fn name_text(n: &X509NameRef) -> String {
    let parts: Vec<String> = n
        .entries()
        .map(|e| {
            let k = e.object().nid().short_name().unwrap_or("?");
            let v = e.data().to_string().unwrap_or_default();
            format!("{}={}", k, v)
        })
        .collect();
    parts.join(", ")
}

fn common_name(n: &X509NameRef) -> Option<String> {
    n.entries_by_nid(Nid::COMMONNAME).next().and_then(|e| e.data().to_string().ok())
}

pub fn fingerprint(cert: &X509Ref) -> Result<String> {
    Ok(hex::encode(cert.digest(MessageDigest::sha256())?))
}

/// Issued by itself: issuer = subject, and the signature checks out with its
/// own key. (X509_check_issued would also ask for the CA key usage.)
fn self_signed(cert: &X509Ref) -> bool {
    let same = matches!((cert.subject_name().to_der(), cert.issuer_name().to_der()), (Ok(a), Ok(b)) if a == b);
    same && cert.public_key().and_then(|k| cert.verify(&k)).unwrap_or(false)
}

/// What a manifest, a log line or `verify` says about a certificate.
pub fn describe(cert: &X509Ref) -> Result<Value> {
    Ok(json!({
        "subject": name_text(cert.subject_name()),
        "issuer": name_text(cert.issuer_name()),
        "self_signed": self_signed(cert),
        "not_after": cert.not_after().to_string(),
        "certificate_sha256": fingerprint(cert)?,
    }))
}

impl Signer {
    /// A PKCS#12 file (.p12/.pfx), or PEM: the certificate (and any chain)
    /// with its key in the same file or in `key`. `password` is asked for
    /// only if the file needs one.
    pub fn load(path: &Path, key: Option<&Path>, password: &mut dyn FnMut() -> Result<String>) -> Result<Signer> {
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let pem = data.windows(11).any(|w| w == b"-----BEGIN ");
        let s = if !pem {
            let p12 = Pkcs12::from_der(&data).with_context(|| format!("{}: not a PKCS#12 or PEM file", path.display()))?;
            let parsed = match p12.parse2("") {
                Ok(p) => p,
                Err(_) => p12.parse2(&password()?).map_err(|_| anyhow!("{}: wrong password", path.display()))?,
            };
            let (Some(key), Some(cert)) = (parsed.pkey, parsed.cert) else { bail!("{}: no private key and certificate", path.display()) };
            Signer { key, cert, chain: parsed.ca.unwrap_or(Stack::new()?) }
        } else {
            let mut certs = X509::stack_from_pem(&data)?;
            if certs.is_empty() {
                bail!("{}: no certificate", path.display());
            }
            let cert = certs.remove(0);
            let kdata = match key {
                Some(k) => std::fs::read(k).with_context(|| format!("reading {}", k.display()))?,
                None => data.clone(),
            };
            let pkey = match PKey::private_key_from_pem(&kdata) {
                Ok(k) => k,
                Err(_) => PKey::private_key_from_pem_passphrase(&kdata, password()?.as_bytes()).map_err(|_| anyhow!("no private key, or wrong password"))?,
            };
            let mut chain = Stack::new()?;
            for c in certs {
                chain.push(c)?;
            }
            Signer { key: pkey, cert, chain }
        };
        if !s.cert.public_key()?.public_eq(&s.key) {
            bail!("{}: the private key does not belong to the certificate", path.display());
        }
        Ok(s)
    }

    pub fn certificate(&self) -> &X509Ref {
        &self.cert
    }

    /// Detached PKCS#7 signature over `data` (SHA-256 with RSA or ECDSA).
    fn pkcs7(&self, data: &[u8]) -> Result<Vec<u8>> {
        let flags = Pkcs7Flags::DETACHED | Pkcs7Flags::BINARY | Pkcs7Flags::NOSMIMECAP;
        Ok(Pkcs7::sign(&self.cert, &self.key, &self.chain, data, flags)?.to_der()?)
    }
}

/// A new RSA-3072 key and a self-signed certificate for signing documents,
/// saved as PKCS#12 at `out` (owner-only permissions) with the public
/// certificate as PEM beside it (`.crt`) for publishing.
pub fn certgen(out: &Path, name: &str, org: Option<&str>, country: Option<&str>, days: u32, password: &str) -> Result<Signer> {
    let key = PKey::from_rsa(Rsa::generate(3072)?)?;
    let mut nb = X509NameBuilder::new()?;
    if let Some(c) = country.filter(|c| !c.is_empty()) {
        nb.append_entry_by_nid(Nid::COUNTRYNAME, c)?;
    }
    if let Some(o) = org.filter(|o| !o.is_empty()) {
        nb.append_entry_by_nid(Nid::ORGANIZATIONNAME, o)?;
    }
    nb.append_entry_by_nid(Nid::COMMONNAME, name)?;
    let subject = nb.build();
    let mut serial = BigNum::new()?;
    serial.rand(127, MsbOption::MAYBE_ZERO, false)?;
    let mut b = X509::builder()?;
    b.set_version(2)?;
    b.set_serial_number(&*Asn1Integer::from_bn(&serial)?)?;
    b.set_subject_name(&subject)?;
    b.set_issuer_name(&subject)?;
    b.set_pubkey(&key)?;
    b.set_not_before(&*Asn1Time::days_from_now(0)?)?;
    b.set_not_after(&*Asn1Time::days_from_now(days)?)?;
    b.append_extension(BasicConstraints::new().critical().build()?)?;
    b.append_extension(KeyUsage::new().critical().digital_signature().non_repudiation().build()?)?;
    // emailProtection as well: what Adobe and OpenSSL's S/MIME checks look for
    b.append_extension(ExtendedKeyUsage::new().email_protection().other(DOCUMENT_SIGNING).build()?)?;
    let ski = SubjectKeyIdentifier::new().build(&b.x509v3_context(None, None))?;
    b.append_extension(ski)?;
    b.sign(&key, MessageDigest::sha256())?;
    let cert = b.build();

    let p12 = Pkcs12::builder().name(name).pkey(&key).cert(&cert).build2(password)?;
    if let Some(d) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(d)?;
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(out).with_context(|| format!("writing {}", out.display()))?.write_all(&p12.to_der()?)?;
    std::fs::write(certificate_path(out), cert.to_pem()?)?;
    Ok(Signer { key, cert, chain: Stack::new()? })
}

// --- writing ------------------------------------------------------------------------------

fn put_name(out: &mut Vec<u8>, name: &[u8]) {
    out.push(b'/');
    for &c in name {
        if b" \t\n\r\x0C()<>[]{}/%#".contains(&c) || !(33..=126).contains(&c) {
            out.extend_from_slice(format!("#{:02X}", c).as_bytes());
        } else {
            out.push(c);
        }
    }
}

/// Serialise a (non-stream) object.
fn put(out: &mut Vec<u8>, o: &Object) -> Result<()> {
    match o {
        Object::Null => out.extend_from_slice(b"null"),
        Object::Boolean(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Object::Integer(i) => out.extend_from_slice(i.to_string().as_bytes()),
        Object::Real(r) => out.extend_from_slice(format!("{}", r).as_bytes()),
        Object::Name(n) => put_name(out, n),
        Object::String(s, _) => {
            out.push(b'<');
            out.extend_from_slice(hex::encode_upper(s).as_bytes());
            out.push(b'>');
        }
        Object::Array(a) => {
            out.push(b'[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b' ');
                }
                put(out, x)?;
            }
            out.push(b']');
        }
        Object::Dictionary(d) => put_dict(out, d)?,
        Object::Reference((n, g)) => out.extend_from_slice(format!("{} {} R", n, g).as_bytes()),
        Object::Stream(_) => bail!("cannot rewrite a stream"),
    }
    Ok(())
}

fn put_dict(out: &mut Vec<u8>, d: &Dictionary) -> Result<()> {
    out.extend_from_slice(b"<<");
    for (k, v) in d.iter() {
        put_name(out, k);
        out.push(b' ');
        put(out, v)?;
        out.push(b' ');
    }
    out.extend_from_slice(b">>");
    Ok(())
}

/// A PDF text string: literal if plain ASCII, else UTF-16BE.
fn text(s: &str) -> String {
    if s.bytes().all(|c| (32..127).contains(&c)) {
        format!("({})", s.replace('\\', "\\\\").replace('(', "\\(").replace(')', "\\)"))
    } else {
        let mut b = vec![0xFE, 0xFF];
        for u in s.encode_utf16() {
            b.extend_from_slice(&u.to_be_bytes());
        }
        format!("<{}>", hex::encode_upper(b))
    }
}

/// "2026-10-28T11:37:00+00:00" -> "D:20261028113700+00'00'"
fn pdf_date(iso: &str) -> String {
    let d: String = iso.chars().take(19).filter(|c| c.is_ascii_digit()).collect();
    format!("D:{}+00'00'", d)
}

/// The offset `startxref` points at, if it is a classic cross-reference table.
fn last_xref(pdf: &[u8]) -> Result<usize> {
    let at = pdf.windows(9).rposition(|w| w == b"startxref").ok_or_else(|| anyhow!("no startxref"))?;
    let num: String = pdf[at + 9..].iter().map(|&c| c as char).skip_while(|c| c.is_whitespace()).take_while(|c| c.is_ascii_digit()).collect();
    let off: usize = num.parse().context("bad startxref")?;
    if !pdf.get(off..).map(|s| s.starts_with(b"xref")).unwrap_or(false) {
        bail!("only PDFs with a classic cross-reference table can be signed");
    }
    Ok(off)
}

/// Sign `pdf`: returns the PDF with the signature appended (the original
/// bytes are an unchanged prefix of the result).
pub fn sign_pdf(pdf: &[u8], s: &Signer, signed_at: &str) -> Result<Vec<u8>> {
    let doc = Document::load_mem(pdf).context("reading PDF")?;
    let prev = last_xref(pdf)?;
    let root = doc.trailer.get(b"Root")?.as_reference()?;
    let mut catalog = doc.get_dictionary(root)?.clone();
    if catalog.has(b"AcroForm") {
        bail!("the PDF already has a form or signature");
    }
    let page = *doc.get_pages().values().next().ok_or_else(|| anyhow!("PDF has no pages"))?;
    let mut page_dict = doc.get_dictionary(page)?.clone();
    let (field, sig) = (doc.max_id + 1, doc.max_id + 2);
    let field_ref = Object::Reference((field, 0));
    match page_dict.get_mut(b"Annots") {
        Ok(Object::Array(a)) => a.push(field_ref.clone()),
        Ok(_) => bail!("page 1 has indirect /Annots"),
        Err(_) => page_dict.set("Annots", Object::Array(vec![field_ref.clone()])),
    }
    let mut acro = Dictionary::new();
    acro.set("Fields", Object::Array(vec![field_ref]));
    acro.set("SigFlags", Object::Integer(3));
    catalog.set("AcroForm", Object::Dictionary(acro));

    // a signature of the same size over anything: room for the real one
    let room = s.pkcs7(b"")?.len() + 512;

    let mut out = pdf.to_vec();
    if out.last() != Some(&b'\n') {
        out.push(b'\n');
    }
    let mut offsets: Vec<(u32, usize)> = Vec::new();
    let mut obj = |out: &mut Vec<u8>, id: u32, body: &[u8]| {
        offsets.push((id, out.len()));
        out.extend_from_slice(format!("{} 0 obj\n", id).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    };
    let mut body = Vec::new();
    put_dict(&mut body, &catalog)?;
    obj(&mut out, root.0, &body);
    body.clear();
    put_dict(&mut body, &page_dict)?;
    obj(&mut out, page.0, &body);
    let widget = format!("<</Type /Annot /Subtype /Widget /FT /Sig /T {} /F 132 /Rect [0 0 0 0] /P {} 0 R /V {} 0 R>>", text(FIELD), page.0, sig);
    obj(&mut out, field, widget.as_bytes());
    let name = common_name(s.cert.subject_name()).unwrap_or_else(|| name_text(s.cert.subject_name()));
    let range_slot = "[0 0000000000 0000000000 0000000000]";
    let head = format!(
        "<</Type /Sig /Filter /Adobe.PPKLite /SubFilter /adbe.pkcs7.detached /Name {} /Reason {} /M {} /ByteRange ",
        text(&name),
        text(REASON),
        text(&pdf_date(signed_at))
    );
    let sig_at = out.len() + format!("{} 0 obj\n", sig).len();
    let range_at = sig_at + head.len();
    let contents_at = range_at + range_slot.len() + " /Contents ".len();
    let mut sig_body = head.into_bytes();
    sig_body.extend_from_slice(range_slot.as_bytes());
    sig_body.extend_from_slice(b" /Contents <");
    sig_body.extend(std::iter::repeat_n(b'0', 2 * room));
    sig_body.extend_from_slice(b">>>");
    obj(&mut out, sig, &sig_body);
    let contents_end = contents_at + 2 * room + 2;

    offsets.sort();
    let xref = out.len();
    out.extend_from_slice(b"xref\n");
    let mut i = 0;
    while i < offsets.len() {
        let mut j = i + 1;
        while j < offsets.len() && offsets[j].0 == offsets[j - 1].0 + 1 {
            j += 1;
        }
        out.extend_from_slice(format!("{} {}\n", offsets[i].0, j - i).as_bytes());
        for (_, off) in &offsets[i..j] {
            out.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
        }
        i = j;
    }
    let mut trailer = Dictionary::new();
    trailer.set("Size", Object::Integer(sig as i64 + 1));
    trailer.set("Root", Object::Reference(root));
    for k in [&b"Info"[..], b"ID"] {
        if let Ok(v) = doc.trailer.get(k) {
            trailer.set(k, v.clone());
        }
    }
    trailer.set("Prev", Object::Integer(prev as i64));
    out.extend_from_slice(b"trailer\n");
    put_dict(&mut out, &trailer)?;
    out.extend_from_slice(format!("\nstartxref\n{}\n%%EOF\n", xref).as_bytes());

    let ranges = format!("[0 {} {} {}]", contents_at, contents_end, out.len() - contents_end);
    if ranges.len() > range_slot.len() {
        bail!("PDF too large to sign");
    }
    out[range_at..range_at + range_slot.len()].copy_from_slice(format!("{:<w$}", ranges, w = range_slot.len()).as_bytes());
    let mut signed = Vec::with_capacity(out.len());
    signed.extend_from_slice(&out[..contents_at]);
    signed.extend_from_slice(&out[contents_end..]);
    let p7 = hex::encode_upper(s.pkcs7(&signed)?);
    if p7.len() > 2 * room {
        bail!("signature larger than the room left for it");
    }
    out[contents_at + 1..contents_at + 1 + p7.len()].copy_from_slice(p7.as_bytes());
    Ok(out)
}

// --- checking -----------------------------------------------------------------------------

/// What `check` found about a PDF's signature.
pub struct SigCheck {
    /// the signature is valid, made with the certificate's key over the bytes
    /// it lists
    pub valid: bool,
    /// those bytes are the whole file but the signature itself
    pub whole_file: bool,
    /// the certificate chains to one of the trusted ones (None: none given)
    pub trusted: Option<bool>,
    pub certificate: Option<X509>,
    pub signed_at: String,
    pub error: Option<String>,
}

fn deref<'a>(doc: &'a Document, o: &'a Object) -> Result<&'a Object> {
    Ok(match o {
        Object::Reference(id) => doc.get_object(*id)?,
        x => x,
    })
}

/// The value dictionary of the first signature field, if any.
fn signature_dict(doc: &Document) -> Result<Option<Dictionary>> {
    let root = doc.trailer.get(b"Root")?.as_reference()?;
    let catalog = doc.get_dictionary(root)?;
    let Ok(acro) = catalog.get(b"AcroForm") else { return Ok(None) };
    let acro = deref(doc, acro)?.as_dict()?;
    let Ok(fields) = acro.get(b"Fields") else { return Ok(None) };
    for f in deref(doc, fields)?.as_array()? {
        let f = deref(doc, f)?.as_dict()?;
        if f.get(b"FT").and_then(|t| t.as_name()).ok() != Some(&b"Sig"[..]) {
            continue;
        }
        if let Ok(v) = f.get(b"V") {
            return Ok(Some(deref(doc, v)?.as_dict()?.clone()));
        }
    }
    Ok(None)
}

/// Check the PDF's digital signature: None if it has none.
pub fn check(pdf: &[u8], trust: &[X509]) -> Result<Option<SigCheck>> {
    let doc = Document::load_mem(pdf).context("reading PDF")?;
    let Some(sig) = signature_dict(&doc)? else { return Ok(None) };
    let mut c = SigCheck { valid: false, whole_file: false, trusted: None, certificate: None, signed_at: String::new(), error: None };
    if let Ok(Object::String(m, _)) = sig.get(b"M") {
        c.signed_at = String::from_utf8_lossy(m).into_owned();
    }
    let ranges: Vec<i64> = sig.get(b"ByteRange")?.as_array()?.iter().map(|x| x.as_i64()).collect::<Result<_, _>>()?;
    let Object::String(contents, _) = sig.get(b"Contents")? else { bail!("signature without /Contents") };
    let ok_ranges = ranges.len() == 4 && ranges.iter().all(|&x| x >= 0) && ranges[0] == 0 && ranges[1] < ranges[2] && (ranges[2] + ranges[3]) as usize <= pdf.len();
    if !ok_ranges {
        c.error = Some(format!("bad /ByteRange {:?}", ranges));
        return Ok(Some(c));
    }
    let (a, b, l) = (ranges[1] as usize, ranges[2] as usize, ranges[3] as usize);
    // the gap is exactly the hex string of /Contents
    let gap = &pdf[a..b];
    c.whole_file = b + l == pdf.len() && gap.len() >= 2 && gap[0] == b'<' && gap[gap.len() - 1] == b'>' && gap[1..gap.len() - 1].iter().all(|x| x.is_ascii_hexdigit());
    let mut data = Vec::with_capacity(a + l);
    data.extend_from_slice(&pdf[..a]);
    data.extend_from_slice(&pdf[b..b + l]);
    let p7 = match Pkcs7::from_der(contents) {
        Ok(p) => p,
        Err(e) => {
            c.error = Some(format!("unreadable signature: {e}"));
            return Ok(Some(c));
        }
    };
    let none = Stack::<X509>::new()?;
    let store: X509Store = X509StoreBuilder::new()?.build();
    match p7.verify(&none, &store, Some(&data), None, Pkcs7Flags::NOVERIFY | Pkcs7Flags::BINARY) {
        Ok(()) => c.valid = true,
        Err(e) => c.error = Some(format!("signature does not match: {e}")),
    }
    let signer = p7.signers(&none, Pkcs7Flags::empty()).ok().and_then(|s| s.iter().next().map(|x| x.to_owned()));
    if let (Some(cert), false) = (&signer, trust.is_empty()) {
        let mut sb = X509StoreBuilder::new()?;
        for t in trust {
            sb.add_cert(t.clone())?;
        }
        let store = sb.build();
        let mut chain = Stack::new()?;
        if let Some(certs) = p7.signed().and_then(|s| s.certificates()) {
            for x in certs {
                chain.push(x.to_owned())?;
            }
        }
        let mut ctx = X509StoreContext::new()?;
        c.trusted = Some(ctx.init(&store, cert, &chain, |x| x.verify_cert())?);
    }
    c.certificate = signer;
    Ok(Some(c))
}

/// Certificates from PEM (one or more) or DER files.
pub fn load_certificates(paths: &[PathBuf]) -> Result<Vec<X509>> {
    let mut out = Vec::new();
    for p in paths {
        let data = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
        if data.windows(11).any(|w| w == b"-----BEGIN ") {
            out.extend(X509::stack_from_pem(&data)?);
        } else {
            out.push(X509::from_der(&data).with_context(|| format!("{}: not a certificate", p.display()))?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_pdf() -> Vec<u8> {
        let mut m = opencv::core::Mat::new_rows_cols_with_default(40, 30, opencv::core::CV_8UC3, opencv::core::Scalar::all(200.0)).unwrap();
        opencv::imgproc::rectangle_points(&mut m, opencv::core::Point::new(5, 5), opencv::core::Point::new(20, 30), opencv::core::Scalar::all(0.0), -1, opencv::imgproc::LINE_8, 0).unwrap();
        crate::io::build_pdf(&[(crate::io::PageImage::Pixels(&m), None, None, 0)]).unwrap()
    }

    #[test]
    fn sign_and_check() {
        let dir = std::env::temp_dir().join(format!("rfpii-sign-{}", std::process::id()));
        let p12 = dir.join("signer.p12");
        let s = certgen(&p12, "Test Returning Officer", Some("Example"), Some("KE"), 30, "pw").unwrap();
        assert_eq!(describe(s.certificate()).unwrap()["self_signed"], true);
        let again = Signer::load(&p12, None, &mut || Ok("pw".into())).unwrap();
        assert_eq!(fingerprint(again.certificate()).unwrap(), fingerprint(s.certificate()).unwrap());
        assert!(Signer::load(&p12, None, &mut || Ok("wrong".into())).is_err());

        let pdf = tiny_pdf();
        let signed = sign_pdf(&pdf, &s, "2026-10-28T11:37:00+00:00").unwrap();
        assert!(signed.starts_with(&pdf));
        // the images are untouched
        assert_eq!(crate::io::pdf_images(&signed).unwrap(), crate::io::pdf_images(&pdf).unwrap());
        assert!(check(&pdf, &[]).unwrap().is_none());
        let c = check(&signed, &[]).unwrap().unwrap();
        assert!(c.valid && c.whole_file && c.trusted.is_none(), "{:?}", c.error);
        assert_eq!(c.signed_at, "D:20261028113700+00'00'");
        let trusted = load_certificates(&[certificate_path(&p12)]).unwrap();
        assert_eq!(check(&signed, &trusted).unwrap().unwrap().trusted, Some(true));
        let other = certgen(&dir.join("other.p12"), "Someone Else", None, None, 30, "x").unwrap();
        assert_eq!(check(&signed, &[other.certificate().to_owned()]).unwrap().unwrap().trusted, Some(false));

        // any change to the signed bytes breaks it
        let mut bad = signed.clone();
        let i = pdf.len() / 2;
        bad[i] ^= 1;
        assert!(!check(&bad, &[]).ok().flatten().map(|c| c.valid).unwrap_or(false));
        // bytes appended after signing are not covered
        let mut more = signed.clone();
        more.extend_from_slice(b"% more\n");
        let c = check(&more, &[]).unwrap().unwrap();
        assert!(c.valid && !c.whole_file);
        // signing twice is refused
        assert!(sign_pdf(&signed, &s, "2026-10-28T11:37:00+00:00").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
