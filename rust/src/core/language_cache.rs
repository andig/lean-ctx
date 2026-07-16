//! Auto-allow reads of dependency source in language caches (#899).
//!
//! A path like `~/go/pkg/mod/github.com/foo@v1/bar.go` or
//! `~/.cargo/registry/src/…/lib.rs` fails the path jail, which used to push
//! agents to `sed` via `ctx_shell` — losing caching, compression and anchored
//! edits exactly where files are largest. This module recognises those paths and
//! registers the cache as a **session read-only root** in [`super::pathjail`], so
//! the read works with no config edit and writes stay denied.

use std::path::{Path, PathBuf};

use super::pathjail::{
    canonicalize_secure, expand_user_path, jail_path_with_roots, register_session_read_only_root,
};

/// The canonical root of a language cache, straight from the toolchain. `None`
/// when the toolchain is absent or names nothing.
type CanonicalRoot = fn() -> Option<PathBuf>;

/// Known language-cache markers: (path substring, human label, config example,
/// canonical-root resolver). Single source of truth shared by
/// [`detected_cache_hint`] (the jail-error suggestion) and
/// [`language_cache_access`] (the auto-allow), so the two never drift.
///
/// The resolver is `Some` only for languages whose canonical root is cheap and
/// deterministic to resolve (Go, Rust). Those get the pass-through path: root
/// from the toolchain, registered and retried inline. `None` languages
/// (site-packages, Maven, …) have no single machine-wide root worth probing, so
/// they keep the conservative pattern-match + retry path.
const LANGUAGE_CACHE_PATTERNS: &[(&str, &str, &str, Option<CanonicalRoot>)] = &[
    (
        "/go/pkg/mod/",
        "Go module cache",
        "~/go/pkg/mod",
        Some(go_mod_cache),
    ),
    (
        "/.cargo/registry/",
        "Rust crate registry",
        "~/.cargo/registry",
        Some(cargo_registry),
    ),
    (
        "/site-packages/",
        "Python site-packages",
        "<venv>/lib/pythonX.Y/site-packages",
        None,
    ),
    (
        "/node_modules/",
        "Node modules",
        "<project>/node_modules",
        None,
    ),
    (
        "/.m2/repository/",
        "Maven local repository",
        "~/.m2/repository",
        None,
    ),
    ("/.gradle/caches/", "Gradle cache", "~/.gradle/caches", None),
    (
        "/.nuget/packages/",
        "NuGet package cache",
        "~/.nuget/packages",
        None,
    ),
];

/// `$GOMODCACHE`, else whatever `go env GOMODCACHE` reports (~5ms).
fn go_mod_cache() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("GOMODCACHE")
        && !v.trim().is_empty()
    {
        return Some(PathBuf::from(v.trim()));
    }
    // ponytail: only the subprocess is memoized (once per process); `go env`
    // reads config files, never the network, so it needs no timeout.
    static PROBED: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    PROBED
        .get_or_init(|| probe_toolchain_path("go", &["env", "GOMODCACHE"]))
        .clone()
}

/// `$CARGO_HOME/registry`, defaulting to `~/.cargo/registry`.
///
/// ponytail: no subprocess — cargo has no `home` subcommand, and `$CARGO_HOME`
/// (default `~/.cargo`) is the exact rule cargo itself applies.
fn cargo_registry() -> Option<PathBuf> {
    let home = std::env::var("CARGO_HOME")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|v| PathBuf::from(v.trim()))
        .or_else(|| dirs::home_dir().map(|h| h.join(".cargo")))?;
    Some(home.join("registry"))
}

/// Run `bin args…` and read a single path off stdout. `None` on a missing
/// toolchain, a non-zero exit, or empty output.
fn probe_toolchain_path(bin: &str, args: &[&str]) -> Option<PathBuf> {
    let out = std::process::Command::new(bin).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!path.is_empty()).then(|| PathBuf::from(path))
}

/// Detect well-known language cache paths and return a targeted hint. Used by
/// jail callers that don't auto-allow (e.g. batch reads); the single-path
/// ctx_read flow instead calls [`language_cache_access`].
pub(crate) fn detected_cache_hint(candidate: &Path) -> Option<String> {
    let s = candidate.to_string_lossy();
    for &(pattern, name, example, _) in LANGUAGE_CACHE_PATTERNS {
        if s.contains(pattern) {
            return Some(format!(
                ". Detected {name} — add read_only_roots = [\"{example}\"] to \
                 ~/.config/lean-ctx/config.toml for cached, compressed reads without write access"
            ));
        }
    }
    None
}

/// If `candidate` sits inside a known language cache, return `(label, root,
/// resolver)` where `root` is the path truncated at the end of the marker
/// directory (no trailing slash).
fn detect_language_cache(
    candidate: &Path,
) -> Option<(&'static str, PathBuf, Option<CanonicalRoot>)> {
    let s = candidate.to_string_lossy().replace('\\', "/");
    for &(marker, label, _, resolver) in LANGUAGE_CACHE_PATTERNS {
        if let Some(idx) = s.find(marker) {
            let end = idx + marker.len() - 1; // keep the marker dir, drop trailing '/'
            return Some((label, PathBuf::from(&s[..end]), resolver));
        }
    }
    None
}

/// What [`language_cache_access`] decided for a jail-rejected path.
pub enum CacheAccess {
    /// Option B: the toolchain's canonical root is now a session read-only root
    /// and the jail re-ran clean. The caller may serve this read immediately.
    PassThrough(PathBuf),
    /// Option A: the marker-derived root is now a session read-only root, but
    /// this call must still fail closed with this message; the agent's retry
    /// resolves.
    Retry(String),
}

/// Auto-allow a jail-rejected path that is dependency source in a language
/// cache, read-only, for this session. Which of the two strategies applies is a
/// property of the language, not a fallback chain:
///
/// - **Option B — Go, Rust.** The toolchain names the canonical root cheaply
///   (`go env GOMODCACHE`, `$CARGO_HOME/registry`), so that root is registered
///   and the jail re-run inline: [`CacheAccess::PassThrough`], zero round-trips.
///   Because the authority is the toolchain and not the path, a lookalike such
///   as `/tmp/evil/go/pkg/mod/x.go` sits outside the real cache and registers
///   nothing — the jail error stands. This is why B never falls back to A: A
///   would register exactly the spoofed root B refused.
/// - **Option A — site-packages, node_modules, Maven, Gradle, NuGet.** No single
///   deterministic root worth probing (a venv or `node_modules` is per-project),
///   so the marker-derived root is registered and the caller fails closed:
///   [`CacheAccess::Retry`].
///
/// `None` means no auto-allow: not a cache path, a B-language lookalike, or a
/// jail that rejects even with the root registered. Either strategy grants
/// **read only** — session roots feed `read_only_roots_from_env_and_config`,
/// which `enforce_writable` also consults, so `ctx_patch` into a cache is denied.
pub fn language_cache_access(
    candidate: &Path,
    jail_root: &Path,
    extra_roots: &[String],
) -> Option<CacheAccess> {
    let (label, marker_root, resolver) = detect_language_cache(candidate)?;

    let Some(resolve) = resolver else {
        // Option A. A repeat call means the root is registered and the jail
        // still says no, so there is nothing left for the retry hint to fix.
        if !register_session_read_only_root(&marker_root) {
            return None;
        }
        return Some(CacheAccess::Retry(format!(
            "Auto-detected {label} at {} — added as a read-only root for this session. \
             Retry the read.",
            marker_root.display()
        )));
    };

    // Option B: trust the toolchain's root, and only when the path is really under it.
    let root = canonicalize_secure(&expand_user_path(&resolve()?.to_string_lossy()));
    if !canonicalize_secure(candidate).starts_with(&root) {
        return None;
    }
    if register_session_read_only_root(&root) {
        tracing::info!(
            "auto-allowed {label} at {} as a read-only root for this session (#899)",
            root.display()
        );
    }
    jail_path_with_roots(candidate, jail_root, extra_roots)
        .ok()
        .map(CacheAccess::PassThrough)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::pathjail::{enforce_writable, is_read_only_path};

    #[test]
    fn detected_cache_hint_recognizes_go_cargo_python() {
        let go = detected_cache_hint(Path::new("/Users/x/go/pkg/mod/github.com/foo/bar/main.go"));
        assert!(go.is_some(), "Go module cache should be detected");
        assert!(go.unwrap().contains("Go module cache"));

        let cargo = detected_cache_hint(Path::new(
            "/home/x/.cargo/registry/src/crates.io/serde-1.0/lib.rs",
        ));
        assert!(cargo.is_some(), "Rust cargo registry should be detected");
        assert!(cargo.unwrap().contains("Rust crate registry"));

        let py = detected_cache_hint(Path::new(
            "/usr/lib/python3.12/site-packages/requests/api.py",
        ));
        assert!(py.is_some(), "Python site-packages should be detected");

        let normal = detected_cache_hint(Path::new("/home/x/projects/myapp/src/main.rs"));
        assert!(normal.is_none(), "Normal project path should not match");
    }

    #[test]
    fn detect_cache_root_extracts_marker_dir() {
        // (path, label, marker-derived root, has a canonical resolver = option B)
        let cases = [
            (
                "/Users/x/go/pkg/mod/github.com/foo/bar@v1.2.3/baz.go",
                "Go module cache",
                "/Users/x/go/pkg/mod",
                true,
            ),
            (
                "/home/u/.cargo/registry/src/index-abc/serde-1.0/src/lib.rs",
                "Rust crate registry",
                "/home/u/.cargo/registry",
                true,
            ),
            (
                "/opt/venv/lib/python3.12/site-packages/requests/api.py",
                "Python site-packages",
                "/opt/venv/lib/python3.12/site-packages",
                false,
            ),
            (
                "/w/app/node_modules/react/index.js",
                "Node modules",
                "/w/app/node_modules",
                false,
            ),
        ];
        for (path, want_label, want_root, want_resolver) in cases {
            let (label, root, resolver) = detect_language_cache(Path::new(path))
                .unwrap_or_else(|| panic!("expected cache match for {path}"));
            assert_eq!(label, want_label, "label for {path}");
            assert_eq!(root, PathBuf::from(want_root), "root for {path}");
            assert_eq!(
                resolver.is_some(),
                want_resolver,
                "option B eligibility for {path}"
            );
        }
        assert!(
            detect_language_cache(Path::new("/home/u/proj/src/main.rs")).is_none(),
            "a normal project path is not a cache"
        );
    }

    /// The core #899 guarantee: once a detected cache root is registered, a path
    /// under it *reads* (jail resolves) but never *writes* (enforce_writable
    /// denies), and registration is idempotent.
    #[cfg(not(feature = "no-jail"))]
    #[test]
    fn registered_cache_root_reads_allow_writes_deny() {
        let _iso = crate::core::data_dir::isolated_data_dir();

        let tmp = tempfile::tempdir().unwrap();
        // A fake Go module cache so detect_language_cache matches the path.
        let dep = tmp.path().join("go/pkg/mod/example.com/lib@v1");
        std::fs::create_dir_all(&dep).unwrap();
        let file = dep.join("lib.go");
        std::fs::write(&file, "package lib").unwrap();

        // A project jail that does NOT contain the cache.
        let project = tmp.path().join("project");
        std::fs::create_dir_all(&project).unwrap();

        // Before registration: the read escapes the jail.
        assert!(jail_path_with_roots(&file, &project, &[]).is_err());

        // Register the detected root; the second call is a no-op.
        let (_, root, _) = detect_language_cache(&file).expect("cache match");
        assert!(
            register_session_read_only_root(&root),
            "first register is new"
        );
        assert!(
            !register_session_read_only_root(&root),
            "re-register is a no-op"
        );

        // After: the read resolves, but writes are denied (read-only tier).
        assert!(
            jail_path_with_roots(&file, &project, &[]).is_ok(),
            "registered cache root must be readable"
        );
        assert!(is_read_only_path(&file), "cache file is read-only");
        assert!(
            enforce_writable(&file).is_err(),
            "writes into the cache root must be denied"
        );
    }

    /// Option B: a Go module cache read passes through on the *first* call — the
    /// toolchain names the root, so there is no retry round-trip — and stays
    /// read-only.
    #[cfg(not(feature = "no-jail"))]
    #[test]
    fn go_cache_under_canonical_root_passes_through_read_only() {
        let _iso = crate::core::data_dir::isolated_data_dir();

        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("gohome/go/pkg/mod");
        let dep = cache.join("example.com/lib@v1");
        std::fs::create_dir_all(&dep).unwrap();
        let file = dep.join("lib.go");
        std::fs::write(&file, "package lib").unwrap();
        let project = tmp.path().join("project");
        std::fs::create_dir_all(&project).unwrap();

        // Stand in for `go env GOMODCACHE`, so the test never spawns a toolchain.
        crate::test_env::set_var("GOMODCACHE", &cache);
        let access = language_cache_access(&file, &project, &[]);
        crate::test_env::remove_var("GOMODCACHE");

        let Some(CacheAccess::PassThrough(jailed)) = access else {
            panic!("a file under GOMODCACHE must pass through on the first call");
        };
        assert_eq!(jailed, canonicalize_secure(&file));
        assert!(
            enforce_writable(&file).is_err(),
            "pass-through grants read, never write"
        );
    }

    /// Option B's safety property: the toolchain, not the path string, is the
    /// authority. A path that merely *looks* like a Go cache is refused, and must
    /// not fall back to option A — which would register the spoofed root.
    #[cfg(not(feature = "no-jail"))]
    #[test]
    fn go_cache_lookalike_outside_canonical_root_is_refused() {
        let _iso = crate::core::data_dir::isolated_data_dir();

        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("real/go/pkg/mod");
        std::fs::create_dir_all(&real).unwrap();
        // Same marker, different tree: attacker-chosen, not the toolchain's.
        let evil = tmp.path().join("evil/go/pkg/mod/example.com/lib@v1");
        std::fs::create_dir_all(&evil).unwrap();
        let file = evil.join("lib.go");
        std::fs::write(&file, "package lib").unwrap();
        let project = tmp.path().join("project");
        std::fs::create_dir_all(&project).unwrap();

        crate::test_env::set_var("GOMODCACHE", &real);
        let access = language_cache_access(&file, &project, &[]);
        crate::test_env::remove_var("GOMODCACHE");

        assert!(
            access.is_none(),
            "a cache lookalike outside GOMODCACHE must not be auto-allowed"
        );
        assert!(
            jail_path_with_roots(&file, &project, &[]).is_err(),
            "and nothing may have been registered for it"
        );
    }

    /// Option A: languages with no deterministic root (here a venv's
    /// site-packages) register the marker-derived root and fail closed once; the
    /// agent's retry then resolves.
    #[cfg(not(feature = "no-jail"))]
    #[test]
    fn site_packages_registers_then_retry_resolves() {
        let _iso = crate::core::data_dir::isolated_data_dir();

        let tmp = tempfile::tempdir().unwrap();
        let dep = tmp
            .path()
            .join("venv/lib/python3.12/site-packages/requests");
        std::fs::create_dir_all(&dep).unwrap();
        let file = dep.join("api.py");
        std::fs::write(&file, "def get(): ...").unwrap();
        let project = tmp.path().join("project");
        std::fs::create_dir_all(&project).unwrap();

        let Some(CacheAccess::Retry(hint)) = language_cache_access(&file, &project, &[]) else {
            panic!("site-packages has no canonical root, so it must fail closed with a hint");
        };
        assert!(hint.contains("Python site-packages"), "hint: {hint}");
        assert!(hint.contains("Retry the read"), "hint: {hint}");

        // The retry resolves, and the cache stays read-only.
        assert!(
            jail_path_with_roots(&file, &project, &[]).is_ok(),
            "the registered root makes the retry resolve"
        );
        assert!(enforce_writable(&file).is_err(), "writes stay denied");
    }

    #[test]
    fn cargo_registry_follows_cargo_home() {
        let _iso = crate::core::data_dir::isolated_data_dir();

        crate::test_env::set_var("CARGO_HOME", "/tmp/cargo-home-fixture");
        let from_env = cargo_registry();
        crate::test_env::remove_var("CARGO_HOME");

        assert_eq!(
            from_env,
            Some(PathBuf::from("/tmp/cargo-home-fixture/registry")),
            "$CARGO_HOME wins, and no subprocess is involved"
        );
        assert_eq!(
            cargo_registry(),
            dirs::home_dir().map(|h| h.join(".cargo/registry")),
            "unset CARGO_HOME falls back to ~/.cargo, cargo's own default"
        );
    }
}
