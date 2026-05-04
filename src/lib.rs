use std::fmt;
use std::hash::Hash;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use moka::sync::Cache;

// ---------------------------------------------------------------------------
// Fingerprint trait + impls
// ---------------------------------------------------------------------------

/// Cheaply identifies whether a file's content has changed since the last parse.
pub trait Fingerprint: Send + Sync + 'static {
    /// Opaque stamp that can be compared for equality and hashed.
    type Stamp: Eq + Hash + Clone + Send + Sync + fmt::Debug + 'static;

    /// Compute the current stamp for `path`. Returns `Err` if the file is
    /// unreadable (missing, permissions, etc.) — the cache treats this as a miss
    /// that produces the caller's error.
    fn stamp(&self, path: &Path) -> io::Result<Self::Stamp>;
}

// ---------------------------------------------------------------------------
// MtimeStamp — serialization-ready replacement for SystemTime
// ---------------------------------------------------------------------------

/// Seconds + nanoseconds since UNIX epoch. Negative `secs` for pre-epoch times.
///
/// Uses floor semantics: `nanos` is always non-negative and `secs` is the
/// largest integer ≤ the true value. For example, 0.5 seconds before epoch
/// is `{ secs: -1, nanos: 500_000_000 }`, representing −1 + 0.5 = −0.5.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(any(test, feature = "persist"), derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MtimeStamp {
    pub secs: i64,
    pub nanos: u32,
}

impl From<SystemTime> for MtimeStamp {
    fn from(t: SystemTime) -> Self {
        match t.duration_since(SystemTime::UNIX_EPOCH) {
            Ok(d) => MtimeStamp {
                secs: d.as_secs() as i64,
                nanos: d.subsec_nanos(),
            },
            Err(e) => {
                let d = e.duration();
                let sub = d.subsec_nanos();
                if sub == 0 {
                    MtimeStamp {
                        secs: -(d.as_secs() as i64),
                        nanos: 0,
                    }
                } else {
                    MtimeStamp {
                        secs: -(d.as_secs() as i64) - 1,
                        nanos: 1_000_000_000 - sub,
                    }
                }
            }
        }
    }
}

impl MtimeStamp {
    /// Convert back to `SystemTime`. Lossless roundtrip with `From<SystemTime>`.
    pub fn to_system_time(self) -> SystemTime {
        if self.secs >= 0 {
            SystemTime::UNIX_EPOCH + Duration::new(self.secs as u64, self.nanos)
        } else if self.nanos == 0 {
            SystemTime::UNIX_EPOCH - Duration::new((-self.secs) as u64, 0)
        } else {
            SystemTime::UNIX_EPOCH
                - Duration::new((-self.secs - 1) as u64, 1_000_000_000 - self.nanos)
        }
    }
}

/// Compares `mtime` from filesystem metadata. Cheap (one syscall), but can
/// miss edits that land within the same second on coarse-grained filesystems,
/// and reports false changes after `git clone` or `cargo` resets mtimes.
#[derive(Debug, Clone, Copy, Default)]
pub struct MtimeFingerprint;

impl Fingerprint for MtimeFingerprint {
    type Stamp = MtimeStamp;

    fn stamp(&self, path: &Path) -> io::Result<Self::Stamp> {
        let mtime = std::fs::metadata(path)?.modified()?;
        Ok(MtimeStamp::from(mtime))
    }
}

/// BLAKE3 hash of the full file content. Robust against mtime resets, but
/// reads the entire file on every check.
#[derive(Debug, Clone, Copy, Default)]
pub struct ContentHashFingerprint;

/// 32-byte BLAKE3 digest, wrapped for trait impls.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(any(test, feature = "persist"), derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Blake3Stamp(pub [u8; 32]);

impl Fingerprint for ContentHashFingerprint {
    type Stamp = Blake3Stamp;

    fn stamp(&self, path: &Path) -> io::Result<Self::Stamp> {
        let bytes = std::fs::read(path)?;
        Ok(Blake3Stamp(*blake3::hash(&bytes).as_bytes()))
    }
}

// ---------------------------------------------------------------------------
// Cache entry (stored inside moka)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Entry<T, S> {
    stamp: S,
    value: T,
}

// ---------------------------------------------------------------------------
// Core cache
// ---------------------------------------------------------------------------

/// Mtime-gated (or content-hash-gated) file parse cache.
///
/// On `get`, the fingerprint of the file is checked. If it matches the cached
/// stamp, the cached `T` is returned without re-parsing. On mismatch or cache
/// miss the `parser` closure runs and the result is stored.
///
/// Backed by `moka::sync::Cache` with bounded-size LRU eviction.
pub struct FileParseCache<T, F: Fingerprint = MtimeFingerprint> {
    inner: Cache<PathBuf, Entry<T, F::Stamp>>,
    fingerprint: Arc<F>,
    dirty: AtomicBool,
}

impl<T, F> fmt::Debug for FileParseCache<T, F>
where
    T: Clone + Send + Sync + 'static + fmt::Debug,
    F: Fingerprint + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.run_pending_tasks();
        f.debug_struct("FileParseCache")
            .field("entry_count", &self.inner.entry_count())
            .field("fingerprint", &self.fingerprint)
            .finish()
    }
}

impl<T: Clone + Send + Sync + 'static> FileParseCache<T, MtimeFingerprint> {
    /// Create a cache with mtime-based invalidation and room for `max_entries` files.
    pub fn new(max_entries: u64) -> Self {
        Self::with_fingerprint(max_entries, MtimeFingerprint)
    }
}

impl<T, F> FileParseCache<T, F>
where
    T: Clone + Send + Sync + 'static,
    F: Fingerprint,
{
    /// Create a cache with a custom fingerprint strategy.
    pub fn with_fingerprint(max_entries: u64, fingerprint: F) -> Self {
        Self {
            inner: Cache::new(max_entries),
            fingerprint: Arc::new(fingerprint),
            dirty: AtomicBool::new(false),
        }
    }

    /// Return the cached value for `path`, or parse it via `parser` on miss /
    /// fingerprint change.
    ///
    /// If multiple threads call `get` for the same path concurrently and all
    /// miss, each thread runs `parser` independently. The last writer wins in
    /// moka; all callers receive a correct (freshly-parsed) value. This trades
    /// a rare redundant parse for a simpler API — coalescing via
    /// `try_get_with` would require wrapping the caller's error in `Arc`.
    ///
    /// On unreadable files (missing, permissions), the fingerprint stat fails
    /// and returns `Err(E::from(io::Error))` without invoking `parser`.
    pub fn get<E>(
        &self,
        path: &Path,
        parser: impl FnOnce(&Path) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<io::Error> + Send + Sync + 'static,
    {
        let key = path.to_path_buf();
        let current_stamp = self.fingerprint.stamp(path).map_err(E::from)?;

        if let Some(entry) = self.inner.get(&key) {
            if entry.stamp == current_stamp {
                return Ok(entry.value.clone());
            }
        }

        let value = parser(path)?;
        self.inner.insert(
            key,
            Entry {
                stamp: current_stamp,
                value: value.clone(),
            },
        );
        self.dirty.store(true, Ordering::Release);
        Ok(value)
    }

    /// Remove entries where `predicate` returns `true`.
    ///
    /// Not atomic: takes a snapshot via iteration, then invalidates matching
    /// keys one by one. An entry inserted by a concurrent `get` between the
    /// snapshot and the invalidation pass will be missed — it will be caught
    /// on the next `purge_if` call. Invalidated entries become immediately
    /// invisible to `get`, but `len()` reflects the removal only after its
    /// own pending-task flush.
    ///
    /// Allocates O(cache size) for the key snapshot — every key is cloned
    /// into a temporary `Vec` before invalidation begins. Fine for caches
    /// under ~10K entries. For larger caches, prefer moka's built-in
    /// TTL/TTI-based eviction over manual purging.
    pub fn purge_if(&self, predicate: impl Fn(&Path) -> bool) {
        let keys_to_remove: Vec<PathBuf> = self
            .inner
            .iter()
            .filter(|(k, _)| predicate(k))
            .map(|(k, _)| k.as_ref().clone())
            .collect();
        if !keys_to_remove.is_empty() {
            for key in &keys_to_remove {
                self.inner.invalidate(key);
            }
            self.dirty.store(true, Ordering::Release);
        }
    }

    /// Remove all entries.
    pub fn clear(&self) {
        self.inner.invalidate_all();
        self.dirty.store(true, Ordering::Release);
    }

    /// Number of entries currently in the cache.
    ///
    /// Flushes pending bookkeeping before reading the count so the value is
    /// immediately consistent with preceding `get`, `purge_if`, and `clear`
    /// calls. Cost is O(pending operations), not O(1) — typically microseconds
    /// for caches with low write rates, but callers in tight loops should be
    /// aware.
    pub fn len(&self) -> u64 {
        self.inner.run_pending_tasks();
        self.inner.entry_count()
    }

    /// Whether the cache is empty.
    ///
    /// Same consistency and cost as [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Persistence (feature = "persist")
// ---------------------------------------------------------------------------

/// Serialization codec for disk persistence. Not object-safe due to generic
/// methods — pass as `&Fmt` where `Fmt: Format`, not `&dyn Format`.
#[cfg(feature = "persist")]
pub trait Format: Send + Sync {
    fn serialize<T: serde::Serialize>(
        &self,
        value: &T,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>;

    fn deserialize<T: serde::de::DeserializeOwned>(
        &self,
        bytes: &[u8],
    ) -> Result<T, Box<dyn std::error::Error + Send + Sync>>;
}

#[cfg(feature = "persist-bincode")]
#[derive(Debug, Clone, Copy, Default)]
pub struct BincodeFormat;

#[cfg(feature = "persist-bincode")]
impl Format for BincodeFormat {
    fn serialize<T: serde::Serialize>(
        &self,
        value: &T,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        bincode::serialize(value).map_err(|e| e as Box<dyn std::error::Error + Send + Sync>)
    }

    fn deserialize<T: serde::de::DeserializeOwned>(
        &self,
        bytes: &[u8],
    ) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
        bincode::deserialize(bytes).map_err(|e| e as Box<dyn std::error::Error + Send + Sync>)
    }
}

#[cfg(feature = "persist-postcard")]
#[derive(Debug, Clone, Copy, Default)]
pub struct PostcardFormat;

#[cfg(feature = "persist-postcard")]
impl Format for PostcardFormat {
    fn serialize<T: serde::Serialize>(
        &self,
        value: &T,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        postcard::to_allocvec(value)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    fn deserialize<T: serde::de::DeserializeOwned>(
        &self,
        bytes: &[u8],
    ) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
        postcard::from_bytes(bytes)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }
}

#[cfg(feature = "persist")]
const DISK_CACHE_VERSION: u32 = 1;

#[cfg(feature = "persist")]
#[derive(serde::Serialize, serde::Deserialize)]
struct DiskCache<T, S> {
    version: u32,
    entries: Vec<DiskEntry<T, S>>,
}

#[cfg(feature = "persist")]
#[derive(serde::Serialize, serde::Deserialize)]
struct DiskEntry<T, S> {
    path: String,
    stamp: S,
    value: T,
}

#[cfg(feature = "persist")]
#[derive(Debug)]
#[non_exhaustive]
pub enum SaveError {
    Io(io::Error),
    Serialize(Box<dyn std::error::Error + Send + Sync>),
}

#[cfg(feature = "persist")]
impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Serialize(e) => write!(f, "serialization error: {e}"),
        }
    }
}

#[cfg(feature = "persist")]
impl std::error::Error for SaveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Serialize(e) => Some(e.as_ref()),
        }
    }
}

#[cfg(feature = "persist")]
#[derive(Debug)]
#[non_exhaustive]
pub enum LoadError {
    Io(io::Error),
    Deserialize(Box<dyn std::error::Error + Send + Sync>),
    VersionMismatch { disk: u32, expected: u32 },
}

#[cfg(feature = "persist")]
impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Deserialize(e) => write!(f, "deserialization error: {e}"),
            Self::VersionMismatch { disk, expected } => {
                write!(f, "version mismatch: disk={disk}, expected={expected}")
            }
        }
    }
}

#[cfg(feature = "persist")]
impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Deserialize(e) => Some(e.as_ref()),
            Self::VersionMismatch { .. } => None,
        }
    }
}

#[cfg(feature = "persist")]
#[derive(Debug, Clone, Copy)]
pub struct LoadStats {
    pub loaded: u64,
    pub stale: u64,
}

#[cfg(feature = "persist")]
impl<T, F> FileParseCache<T, F>
where
    T: Clone + Send + Sync + 'static,
    F: Fingerprint,
{
    /// Persist the current cache to `path` using `format`.
    ///
    /// Returns `Ok(())` immediately if no entries have changed since the last
    /// save. On failure, the dirty flag is restored so the next `save()` retries.
    ///
    /// **Deferred-insert semantics:** A concurrent `get()` that parses and
    /// inserts during `save()` may or may not be included in this snapshot.
    /// If missed, the insert sets the dirty flag, ensuring the next `save()`
    /// captures it. No insert is ever lost as long as the caller saves again
    /// when entries have changed.
    pub fn save<Fmt: Format>(&self, path: &Path, format: &Fmt) -> Result<(), SaveError>
    where
        T: serde::Serialize,
        F::Stamp: serde::Serialize,
    {
        if !self.dirty.swap(false, Ordering::AcqRel) {
            return Ok(());
        }

        let entries: Vec<DiskEntry<T, F::Stamp>> = self
            .inner
            .iter()
            .map(|(k, entry)| DiskEntry {
                path: k.to_string_lossy().into_owned(),
                stamp: entry.stamp.clone(),
                value: entry.value.clone(),
            })
            .collect();

        let disk = DiskCache {
            version: DISK_CACHE_VERSION,
            entries,
        };

        let bytes = match format.serialize(&disk) {
            Ok(b) => b,
            Err(e) => {
                self.dirty.store(true, Ordering::Release);
                return Err(SaveError::Serialize(e));
            }
        };

        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                self.dirty.store(true, Ordering::Release);
                return Err(SaveError::Io(e));
            }
        }

        if let Err(e) = std::fs::write(path, &bytes) {
            self.dirty.store(true, Ordering::Release);
            return Err(SaveError::Io(e));
        }

        Ok(())
    }

    /// Load cached entries from `path`, dropping entries whose fingerprint no
    /// longer matches the file on disk.
    ///
    /// Stale validation is eager: every loaded entry is re-stamped via the
    /// `Fingerprint` impl. Entries for files that were deleted or modified
    /// since the cache was saved are silently skipped. Returns counts of
    /// loaded vs stale entries.
    ///
    /// Does not set the dirty flag — loaded data already matches disk.
    pub fn load<Fmt: Format>(&self, path: &Path, format: &Fmt) -> Result<LoadStats, LoadError>
    where
        T: serde::de::DeserializeOwned,
        F::Stamp: serde::de::DeserializeOwned,
    {
        let bytes = std::fs::read(path).map_err(LoadError::Io)?;
        let disk: DiskCache<T, F::Stamp> =
            format.deserialize(&bytes).map_err(LoadError::Deserialize)?;

        if disk.version != DISK_CACHE_VERSION {
            return Err(LoadError::VersionMismatch {
                disk: disk.version,
                expected: DISK_CACHE_VERSION,
            });
        }

        let mut loaded = 0u64;
        let mut stale = 0u64;

        for entry in disk.entries {
            let file_path = PathBuf::from(&entry.path);
            let current_stamp = match self.fingerprint.stamp(&file_path) {
                Ok(s) => s,
                Err(_) => {
                    stale += 1;
                    continue;
                }
            };
            if current_stamp != entry.stamp {
                stale += 1;
                continue;
            }
            self.inner.insert(
                file_path,
                Entry {
                    stamp: entry.stamp,
                    value: entry.value,
                },
            );
            loaded += 1;
        }

        Ok(LoadStats { loaded, stale })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write as IoWrite;
    use tempfile::TempDir;

    fn write_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        let p = dir.path().join(name);
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        p
    }

    fn make_cache() -> FileParseCache<Vec<String>> {
        FileParseCache::new(64)
    }

    fn line_parser(path: &Path) -> Result<Vec<String>, io::Error> {
        let text = fs::read_to_string(path)?;
        Ok(text.lines().map(String::from).collect())
    }

    // ----- MtimeStamp roundtrip -----

    #[test]
    fn mtime_stamp_post_epoch_roundtrip() {
        let now = SystemTime::now();
        let stamp = MtimeStamp::from(now);
        assert!(stamp.secs > 0);
        assert_eq!(stamp.to_system_time(), now);
    }

    #[test]
    fn mtime_stamp_pre_epoch_roundtrip() {
        let t = SystemTime::UNIX_EPOCH - Duration::new(1, 500_000_000);
        let stamp = MtimeStamp::from(t);
        assert_eq!(stamp.secs, -2);
        assert_eq!(stamp.nanos, 500_000_000);
        assert_eq!(stamp.to_system_time(), t);

        let t = SystemTime::UNIX_EPOCH - Duration::new(0, 500_000_000);
        let stamp = MtimeStamp::from(t);
        assert_eq!(stamp.secs, -1);
        assert_eq!(stamp.nanos, 500_000_000);
        assert_eq!(stamp.to_system_time(), t);

        let t = SystemTime::UNIX_EPOCH - Duration::from_secs(3);
        let stamp = MtimeStamp::from(t);
        assert_eq!(stamp.secs, -3);
        assert_eq!(stamp.nanos, 0);
        assert_eq!(stamp.to_system_time(), t);

        let stamp = MtimeStamp::from(SystemTime::UNIX_EPOCH);
        assert_eq!(stamp.secs, 0);
        assert_eq!(stamp.nanos, 0);
        assert_eq!(stamp.to_system_time(), SystemTime::UNIX_EPOCH);
    }

    #[test]
    fn mtime_stamp_serde_roundtrip() {
        let cases = [
            MtimeStamp { secs: 1_700_000_000, nanos: 123_456_789 },
            MtimeStamp { secs: -2, nanos: 500_000_000 },
            MtimeStamp { secs: -1, nanos: 500_000_000 },
            MtimeStamp { secs: 0, nanos: 0 },
        ];
        for stamp in &cases {
            let json = serde_json::to_string(stamp).unwrap();
            let back: MtimeStamp = serde_json::from_str(&json).unwrap();
            assert_eq!(*stamp, back, "failed roundtrip for {stamp:?}");
        }
    }

    #[test]
    fn blake3_stamp_serde_roundtrip() {
        let bytes = *blake3::hash(b"hello").as_bytes();
        let stamp = Blake3Stamp(bytes);
        let json = serde_json::to_string(&stamp).unwrap();
        let back: Blake3Stamp = serde_json::from_str(&json).unwrap();
        assert_eq!(stamp, back);
    }

    // ----- basic cache behavior -----

    #[test]
    fn returns_parsed_value_and_caches_it() {
        let tmp = TempDir::new().unwrap();
        let p = write_file(&tmp, "a.txt", "hello\nworld");
        let cache = make_cache();

        let first = cache.get(&p, line_parser).unwrap();
        assert_eq!(first, vec!["hello", "world"]);

        let second = cache.get(&p, line_parser).unwrap();
        assert_eq!(second, first);
    }

    #[test]
    fn len_is_consistent_immediately_after_insert() {
        let tmp = TempDir::new().unwrap();
        let a = write_file(&tmp, "a.txt", "a");
        let b = write_file(&tmp, "b.txt", "b");
        let cache = make_cache();

        assert_eq!(cache.len(), 0);
        cache.get(&a, line_parser).unwrap();
        assert_eq!(cache.len(), 1);
        cache.get(&b, line_parser).unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn missing_file_returns_error() {
        let cache = make_cache();
        let result = cache.get(Path::new("/no/such/file.txt"), line_parser);
        assert!(result.is_err());
    }

    // ----- invalidation -----

    #[test]
    fn reparses_when_mtime_changes() {
        let tmp = TempDir::new().unwrap();
        let p = write_file(&tmp, "a.txt", "v1");
        let cache = make_cache();

        let first = cache.get(&p, line_parser).unwrap();
        assert_eq!(first, vec!["v1"]);

        std::thread::sleep(std::time::Duration::from_millis(1100));
        fs::write(&p, "v2\nv3").unwrap();

        let second = cache.get(&p, line_parser).unwrap();
        assert_eq!(second, vec!["v2", "v3"]);
    }

    // ----- content-hash fingerprint -----

    #[test]
    fn content_hash_detects_same_mtime_different_content() {
        let tmp = TempDir::new().unwrap();
        let p = write_file(&tmp, "a.txt", "original");
        let cache: FileParseCache<Vec<String>, ContentHashFingerprint> =
            FileParseCache::with_fingerprint(64, ContentHashFingerprint);

        let first = cache.get(&p, line_parser).unwrap();
        assert_eq!(first, vec!["original"]);

        fs::write(&p, "changed").unwrap();

        let second = cache.get(&p, line_parser).unwrap();
        assert_eq!(second, vec!["changed"]);
    }

    #[test]
    fn content_hash_skips_reparse_on_identical_content() {
        let tmp = TempDir::new().unwrap();
        let p = write_file(&tmp, "a.txt", "stable");

        use std::sync::atomic::{AtomicU32, Ordering};
        let parse_count = Arc::new(AtomicU32::new(0));

        let cache: FileParseCache<Vec<String>, ContentHashFingerprint> =
            FileParseCache::with_fingerprint(64, ContentHashFingerprint);

        let counter = parse_count.clone();
        let counting_parser = move |path: &Path| -> Result<Vec<String>, io::Error> {
            counter.fetch_add(1, Ordering::Relaxed);
            line_parser(path)
        };

        cache.get(&p, &counting_parser).unwrap();
        assert_eq!(parse_count.load(Ordering::Relaxed), 1);

        std::thread::sleep(std::time::Duration::from_millis(1100));
        fs::write(&p, "stable").unwrap();

        cache.get(&p, &counting_parser).unwrap();
        assert_eq!(parse_count.load(Ordering::Relaxed), 1);
    }

    // ----- purge_if -----

    #[test]
    fn purge_if_removes_matching_entries() {
        let tmp = TempDir::new().unwrap();
        let a = write_file(&tmp, "keep.txt", "a");
        let b = write_file(&tmp, "drop.txt", "b");
        let cache = make_cache();

        cache.get(&a, line_parser).unwrap();
        cache.get(&b, line_parser).unwrap();

        cache.purge_if(|p| p.file_name().map_or(false, |n| n == "drop.txt"));

        // len() flushes pending tasks internally — no explicit drain needed.
        assert_eq!(cache.len(), 1);
    }

    // ----- clear -----

    #[test]
    fn clear_removes_all_entries() {
        let tmp = TempDir::new().unwrap();
        let a = write_file(&tmp, "a.txt", "a");
        let b = write_file(&tmp, "b.txt", "b");
        let cache = make_cache();

        cache.get(&a, line_parser).unwrap();
        cache.get(&b, line_parser).unwrap();

        cache.clear();
        // len() flushes pending tasks internally — no explicit drain needed.
        assert_eq!(cache.len(), 0);
    }

    // ----- user-controlled error type -----

    #[derive(Debug)]
    #[allow(dead_code)]
    enum MyError {
        Io(io::Error),
        Parse(String),
    }

    impl From<io::Error> for MyError {
        fn from(e: io::Error) -> Self {
            MyError::Io(e)
        }
    }

    #[test]
    fn parser_error_propagates_without_caching() {
        let tmp = TempDir::new().unwrap();
        let p = write_file(&tmp, "bad.txt", "not-a-number");
        let cache: FileParseCache<i32> = FileParseCache::new(64);

        let result = cache.get(&p, |path| {
            let text = fs::read_to_string(path).map_err(MyError::Io)?;
            text.trim()
                .parse::<i32>()
                .map_err(|e| MyError::Parse(e.to_string()))
        });

        assert!(matches!(result, Err(MyError::Parse(_))));
        // len() flushes pending tasks internally — no explicit drain needed.
        assert_eq!(cache.len(), 0);
    }
}

// ---------------------------------------------------------------------------
// Persistence tests (bincode)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "persist-bincode"))]
mod persist_tests {
    use super::*;
    use std::fs;
    use std::io::Write as IoWrite;
    use tempfile::TempDir;

    fn write_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        let p = dir.path().join(name);
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        p
    }

    fn make_cache() -> FileParseCache<Vec<String>> {
        FileParseCache::new(64)
    }

    fn line_parser(path: &Path) -> Result<Vec<String>, io::Error> {
        let text = fs::read_to_string(path)?;
        Ok(text.lines().map(String::from).collect())
    }

    struct FailingFormat;
    impl Format for FailingFormat {
        fn serialize<T: serde::Serialize>(
            &self,
            _: &T,
        ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
            Err("intentional failure".into())
        }
        fn deserialize<T: serde::de::DeserializeOwned>(
            &self,
            _: &[u8],
        ) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
            Err("intentional failure".into())
        }
    }

    #[test]
    fn save_and_load_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let cache_path = tmp.path().join("cache.bin");
        let p = write_file(&tmp, "a.txt", "hello\nworld");

        let cache = make_cache();
        cache.get(&p, line_parser).unwrap();
        cache.save(&cache_path, &BincodeFormat).unwrap();
        assert!(cache_path.exists());

        let cache2 = make_cache();
        let stats = cache2.load(&cache_path, &BincodeFormat).unwrap();
        assert_eq!(stats.loaded, 1);
        assert_eq!(stats.stale, 0);

        let entries = cache2.get(&p, line_parser).unwrap();
        assert_eq!(entries, vec!["hello", "world"]);
    }

    #[test]
    fn load_drops_stale_entries() {
        let tmp = TempDir::new().unwrap();
        let cache_path = tmp.path().join("cache.bin");
        let p = write_file(&tmp, "a.txt", "v1");

        let cache = make_cache();
        cache.get(&p, line_parser).unwrap();
        cache.save(&cache_path, &BincodeFormat).unwrap();

        // Modify the file so mtime changes — entry becomes stale.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        fs::write(&p, "v2").unwrap();

        let cache2 = make_cache();
        let stats = cache2.load(&cache_path, &BincodeFormat).unwrap();
        assert_eq!(stats.loaded, 0);
        assert_eq!(stats.stale, 1);
        assert_eq!(cache2.len(), 0);
    }

    #[test]
    fn load_drops_missing_files() {
        let tmp = TempDir::new().unwrap();
        let cache_path = tmp.path().join("cache.bin");
        let p = write_file(&tmp, "a.txt", "v1");

        let cache = make_cache();
        cache.get(&p, line_parser).unwrap();
        cache.save(&cache_path, &BincodeFormat).unwrap();

        fs::remove_file(&p).unwrap();

        let cache2 = make_cache();
        let stats = cache2.load(&cache_path, &BincodeFormat).unwrap();
        assert_eq!(stats.loaded, 0);
        assert_eq!(stats.stale, 1);
    }

    #[test]
    fn save_noop_when_not_dirty() {
        let tmp = TempDir::new().unwrap();
        let cache_path = tmp.path().join("cache.bin");
        let cache = make_cache();

        cache.save(&cache_path, &BincodeFormat).unwrap();
        assert!(!cache_path.exists());
    }

    #[test]
    fn save_after_load_is_noop() {
        let tmp = TempDir::new().unwrap();
        let cache_path = tmp.path().join("cache.bin");
        let p = write_file(&tmp, "a.txt", "hello");

        let cache = make_cache();
        cache.get(&p, line_parser).unwrap();
        cache.save(&cache_path, &BincodeFormat).unwrap();

        // Load into fresh cache — dirty should remain false.
        let cache2 = make_cache();
        cache2.load(&cache_path, &BincodeFormat).unwrap();

        // Remove the file and try to save — should be noop (not dirty).
        fs::remove_file(&cache_path).unwrap();
        cache2.save(&cache_path, &BincodeFormat).unwrap();
        assert!(!cache_path.exists());
    }

    #[test]
    fn save_restores_dirty_on_failure() {
        let tmp = TempDir::new().unwrap();
        let cache_path = tmp.path().join("cache.bin");
        let p = write_file(&tmp, "a.txt", "hello");
        let cache = make_cache();

        cache.get(&p, line_parser).unwrap();

        let result = cache.save(&cache_path, &FailingFormat);
        assert!(result.is_err());

        // Dirty was restored — real save should now succeed.
        cache.save(&cache_path, &BincodeFormat).unwrap();
        assert!(cache_path.exists());
    }

    #[test]
    fn version_mismatch_returns_error() {
        let tmp = TempDir::new().unwrap();
        let cache_path = tmp.path().join("cache.bin");

        // Write a cache with a bogus version.
        let disk: DiskCache<Vec<String>, MtimeStamp> = DiskCache {
            version: 99,
            entries: vec![],
        };
        let bytes = bincode::serialize(&disk).unwrap();
        fs::write(&cache_path, &bytes).unwrap();

        let cache = make_cache();
        let result = cache.load(&cache_path, &BincodeFormat);
        assert!(matches!(
            result,
            Err(LoadError::VersionMismatch { disk: 99, expected: 1 })
        ));
    }

    #[test]
    fn multiple_entries_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let cache_path = tmp.path().join("cache.bin");
        let a = write_file(&tmp, "a.txt", "alpha");
        let b = write_file(&tmp, "b.txt", "beta\ngamma");

        let cache = make_cache();
        cache.get(&a, line_parser).unwrap();
        cache.get(&b, line_parser).unwrap();
        cache.save(&cache_path, &BincodeFormat).unwrap();

        let cache2 = make_cache();
        let stats = cache2.load(&cache_path, &BincodeFormat).unwrap();
        assert_eq!(stats.loaded, 2);

        assert_eq!(cache2.get(&a, line_parser).unwrap(), vec!["alpha"]);
        assert_eq!(cache2.get(&b, line_parser).unwrap(), vec!["beta", "gamma"]);
    }
}

// ---------------------------------------------------------------------------
// Persistence tests (postcard)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "persist-postcard"))]
mod postcard_tests {
    use super::*;
    use std::fs;
    use std::io::Write as IoWrite;
    use tempfile::TempDir;

    fn write_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
        let p = dir.path().join(name);
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        p
    }

    fn line_parser(path: &Path) -> Result<Vec<String>, io::Error> {
        let text = fs::read_to_string(path)?;
        Ok(text.lines().map(String::from).collect())
    }

    #[test]
    fn postcard_save_and_load_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let cache_path = tmp.path().join("cache.pc");
        let p = write_file(&tmp, "a.txt", "hello\nworld");

        let cache: FileParseCache<Vec<String>> = FileParseCache::new(64);
        cache.get(&p, line_parser).unwrap();
        cache.save(&cache_path, &PostcardFormat).unwrap();

        let cache2: FileParseCache<Vec<String>> = FileParseCache::new(64);
        let stats = cache2.load(&cache_path, &PostcardFormat).unwrap();
        assert_eq!(stats.loaded, 1);
        assert_eq!(stats.stale, 0);

        let entries = cache2.get(&p, line_parser).unwrap();
        assert_eq!(entries, vec!["hello", "world"]);
    }
}
