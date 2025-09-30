# Concurrency and File Lock Fixes

**Date:** September 30, 2025
**Status:** ✅ Complete
**Type:** Bug Fix + Architecture Improvement

## Summary

Fixed critical race condition in concurrent dataset fetching and improved file locking reliability with monotonic time-based timeouts.

## Problems Identified

### 1. Race Condition in Dataset Fetching

With `fetch="always"`, there was a race between "refresh" and "mirror":

**The Race:**

1. Thread A acquires lock, deletes cache file, fetches, **releases lock**
2. Thread A attempts to mirror from cache to requested path
3. Thread B acquires lock, deletes cache file (that Thread A just created!)
4. Thread A tries to complete mirror → source file doesn't exist → OSError

**Impact:** Sporadic failures in concurrent environments, especially with `fetch="always"` policy.

### 2. File Lock Timeout Issues

**Race in test:**

- Test tried to acquire lock before holder thread actually held it
- Second acquire sometimes succeeded, never timing out

**Clock-based timeout:**

- Used `time.time()` (wall-clock) instead of `time.monotonic()`
- System clock jumps could break timeout guarantees
- Not reliable for production use

## Solutions Implemented

### A. Atomic Fetch + Mirror Operations

**File:** `src/glassalpha/pipeline/audit.py`

Made fetch + mirror atomic by moving all operations inside the lock:

```python
with file_lock(lock_path):
    # 1. Respect fetch policy inside lock
    if self.config.data.fetch == "always":
        try:
            final_cache_path.unlink()
        except FileNotFoundError:
            pass

    # 2. Fetch if needed
    if not final_cache_path.exists():
        logger.info(f"Fetching dataset {spec.key} into cache")
        produced = spec.fetch_fn()

        # 3. Atomic publish to cache via temp + replace
        if produced.resolve() != final_cache_path:
            tmp = final_cache_path.with_suffix(".tmp")
            _retry_io(lambda: shutil.move(str(produced), str(tmp)))
            _retry_io(lambda: os.replace(str(tmp), str(final_cache_path)))

    # 4. Mirror inside lock (KEY FIX - prevents race)
    if requested_path.resolve() != final_cache_path:
        def _mirror():
            if requested_path.exists():
                return  # Idempotent
            try:
                os.link(str(final_cache_path), str(requested_path))
            except OSError:
                shutil.copy2(str(final_cache_path), str(requested_path))

        _retry_io(_mirror)
```

**Key Improvements:**

1. **Mirror inside lock** - No other thread can delete cache during mirroring
2. **Retry wrapper** - Handles transient filesystem issues with exponential backoff
3. **Atomic operations** - Temp file + `os.replace()` for cache publish
4. **Idempotent checks** - Tolerate "already exists" scenarios

### B. Robust File Locking

**File:** `src/glassalpha/utils/locks.py`

Fixed timeout mechanism and added observability:

```python
@contextmanager
def file_lock(lock_path: Path, timeout_s: float = 60.0, retry_ms: int = 100):
    """
    Cross-platform file lock using atomic create.
    - Uses monotonic time for reliable timeouts
    - Writes PID for debugging
    - Creates parent directories automatically
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + float(timeout_s)  # KEY FIX: monotonic time
    fd = None

    try:
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)

                # Write PID for observability
                try:
                    os.write(fd, f"{os.getpid()}".encode())
                except OSError:
                    pass

                break  # Lock acquired

            except FileExistsError:
                # Check timeout with monotonic clock
                if timeout_s == 0 or time.monotonic() >= deadline:
                    raise TimeoutError(f"Lock timed out: {lock_path}")

                time.sleep(retry_ms / 1000.0)

        yield

    finally:
        # Best-effort cleanup
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass
```

**Key Improvements:**

1. **`time.monotonic()`** - Immune to system clock adjustments
2. **PID in lock file** - Helps debug who owns stuck locks
3. **Parent directory creation** - Prevents ENOENT on first use
4. **`timeout_s=0`** - Enables non-blocking try-lock pattern
5. **Parameter reorder** - `timeout_s` before `retry_ms` (more commonly used)

### C. Deterministic Lock Tests

**File:** `tests/test_concurrency_fetch.py`

Fixed race condition in test using `threading.Event`:

```python
def test_file_lock_timeout(self, tmp_path):
    """Test that file lock times out appropriately."""
    lock_path = tmp_path / "test_lock"
    locked = threading.Event()  # KEY: synchronization primitive

    def hold_lock():
        with file_lock(lock_path, timeout_s=1.0):
            locked.set()  # Signal lock is held
            time.sleep(2)  # Hold past timeout

    t = threading.Thread(target=hold_lock, daemon=True)
    t.start()

    # KEY: Wait for lock to definitely be held
    assert locked.wait(1.0), "holder thread never acquired the lock"

    # Now this should timeout (deterministically!)
    with pytest.raises(TimeoutError):
        with file_lock(lock_path, timeout_s=0.5, retry_ms=50):
            pass

    t.join()
```

**Why This Works:**

- `Event` removes the race - second acquire waits until first is real
- `tmp_path` avoids `/tmp` symlink issues on macOS
- Test is deterministic and passes consistently

## Additional Hardening

### Retry Logic for Concurrent Directory Creation

**File:** `src/glassalpha/utils/cache_dirs.py`

Added retry mechanism to `ensure_dir_writable()`:

```python
def ensure_dir_writable(path: Path, mode: int = 0o700) -> Path:
    """Ensure a directory exists and is writable with concurrent safety."""
    import time

    path = Path(path).resolve()
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            path.mkdir(parents=True, exist_ok=True)

            # Small delay for filesystem to settle
            if attempt > 0:
                time.sleep(0.01)

            if not path.exists():
                if attempt < max_attempts - 1:
                    time.sleep(0.05)
                    continue
                raise RuntimeError(f"Failed to create directory: {path}")

            # Test writability
            test_file = path / ".write_test"
            test_file.write_text("ok")
            test_file.unlink()

            return path

        except FileNotFoundError as e:
            if attempt < max_attempts - 1:
                time.sleep(0.05 * (attempt + 1))
                continue
            raise RuntimeError(f"Cannot create directory: {path}") from e
```

**Handles:** Transient errors during concurrent directory creation on NFS, Windows, etc.

## Test Results

### All Concurrency Tests Pass

```
tests/test_concurrency_fetch.py::TestConcurrencyFetch::test_concurrent_fetch_same_dataset PASSED
tests/test_concurrency_fetch.py::TestConcurrencyFetch::test_file_lock_timeout PASSED
tests/test_concurrency_fetch.py::TestConcurrencyFetch::test_file_lock_context_manager PASSED
tests/test_concurrency_fetch.py::TestConcurrencyFetch::test_file_lock_multiple_acquisitions PASSED
tests/test_concurrency_fetch.py::TestConcurrencyFetch::test_lock_path_generation PASSED

5 passed in 3.05s
```

### Cache Directory Tests Still Pass

```
tests/test_cache_dirs.py - 13 passed in 0.21s
```

### Concurrent Fetch Validation

- ✅ All 3 threads succeed (return Path objects)
- ✅ Only 1 fetch call happens (lock serializes access)
- ✅ All results point to the same file
- ✅ No sporadic failures or race conditions
- ✅ Works with both `fetch="if_missing"` and `fetch="always"`

## Files Modified

1. **`src/glassalpha/pipeline/audit.py`**

   - Moved mirror operation inside lock
   - Added retry wrapper for I/O operations
   - Atomic file operations with temp + replace

2. **`src/glassalpha/utils/locks.py`**

   - Changed to `time.monotonic()` for timeouts
   - Added PID writing for debugging
   - Parent directory creation
   - Reordered parameters (`timeout_s` first)

3. **`src/glassalpha/utils/cache_dirs.py`**

   - Added retry logic to `ensure_dir_writable()`
   - Better handling of concurrent directory creation

4. **`tests/test_concurrency_fetch.py`**
   - Fixed race in timeout test with `threading.Event`
   - Updated all tests to use `tmp_path`
   - Mock fetcher now creates actual files

## Benefits

| Aspect                 | Before                            | After                            |
| ---------------------- | --------------------------------- | -------------------------------- |
| **Concurrent fetches** | Race condition, sporadic failures | Serialized, atomic, reliable     |
| **Lock timeouts**      | Wall-clock, unreliable            | Monotonic, robust                |
| **Test reliability**   | Flaky due to race                 | Deterministic, always passes     |
| **Observability**      | No info on stuck locks            | PID in lock file                 |
| **Filesystem safety**  | Single attempt                    | Retry logic for transient errors |

## Production Readiness

The dataset fetching system is now:

✅ **Thread-safe** - Multiple threads can safely fetch the same dataset
✅ **Process-safe** - File locks coordinate across processes
✅ **Atomic** - No partial states visible to other threads
✅ **Robust** - Retry logic handles transient filesystem issues
✅ **Observable** - PID in lock files aids debugging
✅ **Testable** - Deterministic tests validate behavior

## Future Enhancements (Optional)

1. **Jitter for hot contention:**

   ```python
   import random
   sleep_time = (retry_ms + random.randint(0, retry_ms)) / 1000.0
   time.sleep(sleep_time)
   ```

2. **Non-blocking try-lock pattern:**

   ```python
   try:
       with file_lock(path, timeout_s=0):  # Already supported!
           # Got lock immediately
           do_work()
   except TimeoutError:
       # Lock is held, skip or queue for later
       pass
   ```

3. **Lock age monitoring:**
   - Check lock file mtime
   - Warn if lock held > threshold
   - Auto-cleanup stale locks (with care!)

## References

- [PEP 418](https://peps.python.org/pep-0418/) - Add monotonic time, performance counter, and process time functions
- [Python threading.Event](https://docs.python.org/3/library/threading.html#threading.Event) - Thread synchronization primitives
- [os.open() flags](https://docs.python.org/3/library/os.html#os.open) - O_CREAT | O_EXCL for atomic creation
