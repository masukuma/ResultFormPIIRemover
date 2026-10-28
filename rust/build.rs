use std::process::Command;

fn main() {
    // PDF rendering with MuPDF: the very library PyMuPDF ships, so that pages
    // render to the same pixels as in the Python version. JPEG coefficient
    // access goes through the libjpeg compiled into it (one libjpeg in the
    // process; OpenCV keeps its own hidden).
    let mupdf = std::env::var("MUPDF_PREFIX").unwrap_or_else(|_| "/opt/mupdf".into());
    cc::Build::new().file("csrc/render.c").include(format!("{}/include", mupdf)).warnings(true).compile("mrender");
    // JPEG coefficient access: a private copy of the static libjpeg(-turbo),
    // every symbol renamed, so that it can't meet MuPDF's libjpeg (which
    // exports the standard names) or OpenCV's.
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let libjpeg = std::env::var("LIBJPEG_STATIC").unwrap_or_else(|_| "/usr/lib/x86_64-linux-gnu/libjpeg.a".into());
    let nm = Command::new("nm").args(["-g", "--defined-only", &libjpeg]).output().expect("nm");
    let mut syms: Vec<String> = String::from_utf8_lossy(&nm.stdout)
        .lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (f.len() == 3).then(|| f[2].to_string())
        })
        .collect();
    syms.sort();
    syms.dedup();
    assert!(syms.iter().any(|s| s == "jpeg_read_coefficients"), "{} is not a libjpeg archive", libjpeg);
    let map: String = syms.iter().map(|s| format!("{} rfpii_{}\n", s, s)).collect();
    std::fs::write(out.join("jpeg.syms"), map).unwrap();
    let hdr: String = syms.iter().map(|s| format!("#define {} rfpii_{}\n", s, s)).collect();
    std::fs::write(out.join("jpeg_prefix.h"), hdr).unwrap();
    let st = Command::new("objcopy")
        .arg(format!("--redefine-syms={}", out.join("jpeg.syms").display()))
        .arg(&libjpeg)
        .arg(out.join("librfpiijpeg.a"))
        .status()
        .expect("objcopy");
    assert!(st.success(), "objcopy failed");
    cc::Build::new().file("csrc/jdct.c").flag("-include").flag(out.join("jpeg_prefix.h").to_str().unwrap()).warnings(true).compile("jdct");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=rfpiijpeg");
    println!("cargo:rerun-if-env-changed=LIBJPEG_STATIC");
    println!("cargo:rustc-link-search=native={}/lib", mupdf);
    println!("cargo:rustc-link-lib=dylib=mupdf");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", mupdf);
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-env-changed=MUPDF_PREFIX");

    // OpenCV 5 lives outside the system library path
    let prefix = std::env::var("OPENCV5_PREFIX").unwrap_or_else(|_| "/opt/opencv5".into());
    println!("cargo:rustc-link-search=native={}/lib", prefix); // ahead of a system OpenCV 4
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", prefix);
    println!("cargo:rerun-if-env-changed=OPENCV5_PREFIX");

    // The code revision, recorded in every manifest (with a mark when the
    // source differs from that commit).
    let git = |args: &[&str]| {
        Command::new("git").args(args).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_default();
    let dirty = git(&["status", "--porcelain", "--", "."]).map(|s| !s.is_empty()).unwrap_or(false);
    let rev = if commit.is_empty() { String::new() } else if dirty { format!("{}+uncommitted-changes", commit) } else { commit };
    println!("cargo:rustc-env=RFPII_COMMIT={}", rev);
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");
}
