"""Pinned official release download with parallel ranges and digest verification."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.request


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    algorithm = hashlib.sha256() if "sha256" in entry else hashlib.sha1()
    if "blob_id" in entry:
        algorithm.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            algorithm.update(block)
    return algorithm.hexdigest() == entry.get("sha256", entry.get("blob_id"))


def open_url(url, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0", **(headers or {})})
    return urllib.request.urlopen(request, timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()
    release = json.loads(Path(__file__).with_name("release.json").read_text())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    required = sum(e["size"] for name, e in release["files"].items()
                   if not (root / name).is_file())
    if shutil.disk_usage(root).free < required + (2 << 30):
        raise RuntimeError("Insufficient disk space for release plus 2 GiB reserve")
    base = f"{args.endpoint.rstrip('/')}/{release['repository']}/resolve/{release['revision']}/"
    state = {"received_bytes": 0, "verified_files": 0, "phase": "download"}
    lock, stop = threading.Lock(), threading.Event()
    started = time.monotonic()

    def emit(event, **fields):
        print(json.dumps(dict(event=event, **fields)), flush=True)

    def heartbeat():
        previous_bytes, previous_time = 0, started
        while not stop.wait(5):
            now = time.monotonic()
            with lock:
                snapshot = dict(state)
            emit("download_progress", **snapshot, elapsed_s=now-started,
                 recent_MB_s=(snapshot["received_bytes"]-previous_bytes)/1e6/(now-previous_time))
            previous_bytes, previous_time = snapshot["received_bytes"], now

    def fetch(item):
        name, entry = item
        destination = root / name
        if verify(destination, entry):
            with lock:
                state["verified_files"] += 1
            emit("file_verified", name=name, reused=True, bytes=entry["size"])
            return
        partial = destination.with_name(destination.name + ".partial")
        url = base + name + "?download=true"
        if entry["size"] > 64 << 20:
            # Separate URLs prevent intermediary caches mixing different ranges.
            chunk_size = 32 << 20
            with partial.open("wb") as target:
                target.truncate(entry["size"])

                def get_range(start):
                    end = min(start + chunk_size, entry["size"]) - 1
                    for attempt in range(4):
                        try:
                            with open_url(url + f"&range_start={start}",
                                          {"Range": f"bytes={start}-{end}"}) as source:
                                expected = f"bytes {start}-{end}/{entry['size']}"
                                if source.status != 206 or source.headers.get("Content-Range") != expected:
                                    raise ValueError("Server did not return the requested byte range")
                                offset = start
                                while block := source.read(1 << 20):
                                    if offset + len(block) > end + 1:
                                        raise ValueError("Oversized byte range")
                                    view = memoryview(block)
                                    while view:
                                        count = os.pwrite(target.fileno(), view, offset)
                                        if count <= 0:
                                            raise OSError("Short pwrite")
                                        offset += count
                                        view = view[count:]
                                    with lock:
                                        state["received_bytes"] += len(block)
                                if offset != end + 1:
                                    raise ValueError("Incomplete byte range")
                            return
                        except Exception as error:
                            emit("range_retry", name=name, start=start, attempt=attempt+1, error=repr(error))
                            if attempt == 3:
                                raise
                            time.sleep(2 ** (attempt + 1))

                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(get_range, range(0, entry["size"], chunk_size)))
        else:
            with open_url(url) as source, partial.open("wb") as target:
                while block := source.read(1 << 20):
                    target.write(block)
                    with lock:
                        state["received_bytes"] += len(block)
        emit("verifying_file", name=name)
        if not verify(partial, entry):
            raise ValueError(f"Digest mismatch for {name}")
        partial.replace(destination)
        with lock:
            state["verified_files"] += 1
        emit("file_verified", name=name, reused=False, bytes=entry["size"])

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, release["files"].items()))
        (root / "verified_release.json").write_text(json.dumps(release, indent=2) + "\n")
        emit("download_complete", repository=release["repository"], revision=release["revision"],
             elapsed_s=time.monotonic()-started, **state)
    finally:
        stop.set()
        thread.join()


if __name__ == "__main__":
    main()
