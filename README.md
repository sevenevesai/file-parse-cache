# file-parse-cache

In-memory cache for apps that repeatedly poll files and reparse them. Stores parsed values keyed by file path; re-parses only when the file's fingerprint (mtime by default, or content hash) changes. Backed by [moka](https://crates.io/crates/moka) for bounded-size LRU eviction — you set a max entry count and stale entries get evicted automatically.

**Not for:** incremental computation engines ([salsa](https://github.com/salsa-rs/salsa)), high-throughput concurrent caches where you'd use [moka](https://crates.io/crates/moka) directly, or build tools that need cargo-style fingerprinting (see [rust-lang/cargo#11682](https://github.com/rust-lang/cargo/issues/11682) for why mtime alone isn't enough in CI). This crate solves the narrow problem of "I stat a file every N seconds and want to skip the parse when nothing changed."

## Usage

```rust
use file_parse_cache::FileParseCache;
use std::io;
use std::path::Path;

// Cache up to 1024 parsed files, evicting least-recently-used.
let cache = FileParseCache::new(1024);

fn parse_config(path: &Path) -> Result<Vec<String>, io::Error> {
    let text = std::fs::read_to_string(path)?;
    Ok(text.lines().map(String::from).collect())
}

// First call parses; subsequent calls return the cached value
// until the file's mtime changes.
let lines = cache.get(Path::new("config.txt"), parse_config)?;
```

For robustness against mtime resets (git clone, CI artifact extraction), use `ContentHashFingerprint`:

```rust
use file_parse_cache::{FileParseCache, ContentHashFingerprint};

let cache: FileParseCache<Vec<String>, ContentHashFingerprint> =
    FileParseCache::with_fingerprint(1024, ContentHashFingerprint);
```

## Persistence

Enable the `persist-bincode` or `persist-postcard` feature to add `save()`/`load()` methods that persist the cache to disk across restarts.

```toml
[dependencies]
file-parse-cache = { version = "0.1", features = ["persist-bincode"] }
```

```rust
use file_parse_cache::{FileParseCache, BincodeFormat};
use std::path::Path;

let cache = FileParseCache::new(1024);

// ... populate via cache.get() calls ...

// Persist to disk. No-op if nothing changed since last save.
cache.save(Path::new("my-cache.bin"), &BincodeFormat)?;

// On next startup, load from disk. Entries whose files changed
// since the save are silently dropped (eager stale validation).
cache.load(Path::new("my-cache.bin"), &BincodeFormat)?;
```

The `persist` feature exposes a `Format` trait if you want a serialization format other than bincode or postcard — implement two methods (`serialize`, `deserialize`) wrapping your chosen codec.

`T` and `Fingerprint::Stamp` must implement `serde::Serialize + DeserializeOwned` for persistence. The built-in `MtimeStamp` and `Blake3Stamp` types do when the `persist` feature is active.

## Save/load semantics

`save()` uses swap-then-restore dirty tracking: a no-op when nothing changed, and the dirty flag is restored on write failure so the next call retries. Concurrent `get()` calls during `save()` may or may not land in the current snapshot — if missed, they're captured by the next save. No insert is ever lost.

`load()` eagerly re-stamps every entry against the file on disk. Stale and missing entries are silently dropped. The return value tells you how many loaded vs how many were stale.

There is no `flush()` that guarantees all concurrent inserts are persisted before returning. If you need that guarantee, stop writing before you save.

## Scope

No async API. No TTL expiry (use moka directly for that). No cache warming or prefetch. Designed to extract cleanly from applications that already have the poll-and-reparse pattern.
