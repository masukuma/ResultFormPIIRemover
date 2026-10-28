use std::process::Command;

fn main() {
    // JPEG coefficient access through the system libjpeg(-turbo)
    cc::Build::new().file("csrc/jdct.c").warnings(true).compile("jdct");
    println!("cargo:rustc-link-lib=jpeg");
    println!("cargo:rerun-if-changed=csrc/jdct.c");

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
