"""Download only the pinned English MTEB HR snapshot used by the public score."""
import argparse
import json
import multiprocessing
from pathlib import Path
import re
import time
import urllib.request
from run_hf_baseline import sha256

REPO = 'vidore/vidore_v3_hr_mteb_format'
REVISION = 'bc7d43d64815ed30f664168c8052106484aba7fd'
FILES = {
    'english-corpus/test-00000-of-00001.parquet': '93b89ac0665e8e1dc3a298bf8f5b4ec7809dec01e2564f40335a8ca1d9e098ad',
    'english-queries/test-00000-of-00001.parquet': '385874527338ca229f2a5634c45513378c8460d123ea2db4758e63c24a3b95be',
    'english-qrels/test-00000-of-00001.parquet': '0d164f3769db821b044ec13a2ed6d321016107f77af7c4f5bdcf5471c46e2c81',
}
SIZES = dict(zip(FILES, (440255343, 22699, 26720)))


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def transfer(url, partial, size, socket_timeout, channel):
    """Isolated worker: parent can terminate even a blocked DNS/TLS/read call."""
    try:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {'User-Agent': 'ColQwen-pinned-HR-downloader/1',
                   'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=socket_timeout) as response:
            if response.status == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                     response.headers.get('Content-Range', ''))
                if not match or int(match[1]) != offset or int(match[3]) != size:
                    raise ValueError('Server returned an inconsistent Content-Range')
            elif response.status == 200:
                offset = 0  # Server ignored Range: restart rather than append.
            else:
                raise ValueError(f'Unexpected HTTP status {response.status}')
            channel.send(('receiving', offset))
            with partial.open('ab' if offset else 'wb') as output:
                while True:
                    block = response.read1(256 * 1024)
                    if not block:
                        break
                    if offset + len(block) > size:
                        raise ValueError('Response exceeds pinned file size')
                    output.write(block)
                    offset += len(block)
                    channel.send(('receiving', offset))
        if offset != size:
            raise ValueError(f'Truncated response: received {offset} of {size} bytes')
        channel.send(('complete', offset))
    except Exception as error:
        channel.send(('error', f'{type(error).__name__}: {error}'))
    finally:
        channel.close()


def attempt(url, partial, size, args, report):
    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=transfer,
                             args=(url, partial, size, args.socket_timeout, sender))
    start = last_data = last_report = time.monotonic()
    received = initial = partial.stat().st_size if partial.exists() else 0
    phase = 'connecting'
    worker.start()
    sender.close()
    report(phase=phase, bytes=received, total_bytes=size, percent=100*received/size,
           elapsed_s=0, no_data_s=0, mb_per_s=0)
    try:
        while True:
            if receiver.poll(min(0.2, args.progress_seconds)):
                try:
                    state, value = receiver.recv()
                except EOFError:
                    raise RuntimeError('Download worker exited without a completion message')
                if state == 'error':
                    raise RuntimeError(value)
                if value != received:
                    if value < received:
                        initial = value
                    received = value
                    last_data = time.monotonic()
                phase = state
                if state == 'complete':
                    report(phase=phase, bytes=received, total_bytes=size, percent=100,
                           elapsed_s=round(time.monotonic()-start, 2), no_data_s=0)
                    return
            now = time.monotonic()
            if now-last_report >= args.progress_seconds:
                report(phase=phase, bytes=received, total_bytes=size,
                       percent=round(100*received/size, 2), elapsed_s=round(now-start, 2),
                       no_data_s=round(now-last_data, 2),
                       mb_per_s=round(max(0, received-initial)/1e6/(now-start), 3))
                last_report = now
            if now-last_data >= args.stall_seconds:
                raise TimeoutError(f'No new bytes for {args.stall_seconds:g} seconds')
            if now-start >= args.attempt_seconds:
                raise TimeoutError(f'Attempt exceeded {args.attempt_seconds:g} seconds')
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=5)
        if worker.is_alive():
            worker.kill()
            worker.join()
        receiver.close()


def download_file(root, name, digest, size, endpoints, args):
    destination = root/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name+'.part')
    if destination.exists():
        emit('verifying_existing', file=name, bytes=destination.stat().st_size)
        if destination.stat().st_size != size or sha256(destination) != digest:
            raise ValueError(f'Existing file differs from pinned asset: {destination}; '
                             'preserved unchanged, use a separate destination')
        emit('verified', file=name, sha256=digest, reused=True)
        return
    for endpoint in endpoints:
        url = f'{endpoint.rstrip("/")}/datasets/{REPO}/resolve/{REVISION}/{name}'
        for number in range(1, args.attempts+1):
            try:
                if partial.exists() and partial.stat().st_size > size:
                    partial.unlink()
                if not partial.exists() or partial.stat().st_size != size:
                    attempt(url, partial, size, args,
                            lambda **row: emit('progress', file=name, endpoint=endpoint,
                                               attempt=number, **row))
                emit('verifying', file=name, bytes=size)
                actual = sha256(partial)
                if actual != digest:
                    partial.unlink()
                    raise ValueError(f'SHA256 mismatch: expected {digest}, received {actual}; '
                                     'discarded corrupt partial download')
                partial.replace(destination)
                emit('verified', file=name, sha256=actual, endpoint=endpoint, reused=False)
                return
            except Exception as error:
                emit('attempt_failed', file=name, endpoint=endpoint, attempt=number,
                     error=str(error), partial_bytes=partial.stat().st_size if partial.exists() else 0)
    raise RuntimeError(f'All download attempts failed for {name}; partial file retained at {partial}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--endpoint', default='https://huggingface.co')
    p.add_argument('--fallback-endpoint', default='https://hf-mirror.com',
                   help='Used after primary attempts fail; empty string disables fallback')
    p.add_argument('--progress-seconds', type=float, default=5)
    p.add_argument('--socket-timeout', type=float, default=30)
    p.add_argument('--stall-seconds', type=float, default=60)
    p.add_argument('--attempt-seconds', type=float, default=1800)
    p.add_argument('--attempts', type=int, default=2, help='Attempts per file per endpoint')
    args = p.parse_args()
    if min(args.progress_seconds, args.socket_timeout, args.stall_seconds,
           args.attempt_seconds, args.attempts) <= 0:
        p.error('Intervals, timeouts and attempt count must be positive')
    endpoints = list(dict.fromkeys(v for v in (args.endpoint, args.fallback_endpoint) if v))
    if not endpoints:
        p.error('At least one endpoint is required')
    start = time.monotonic()
    args.root.mkdir(parents=True, exist_ok=True)
    # A stale success manifest must not survive a failed re-verification.
    (args.root/'manifest.json').unlink(missing_ok=True)
    emit('start', repo=REPO, revision=REVISION, total_bytes=sum(SIZES.values()),
         root=str(args.root.resolve()), endpoints=endpoints,
         progress_seconds=args.progress_seconds, stall_seconds=args.stall_seconds,
         socket_timeout=args.socket_timeout, attempt_seconds=args.attempt_seconds,
         attempts_per_endpoint=args.attempts)
    try:
        for name, expected in FILES.items():
            download_file(args.root, name, expected, SIZES[name], endpoints, args)
    except Exception as error:
        emit('finish', status='failed', error=str(error), elapsed_s=time.monotonic()-start)
        raise SystemExit(1)
    record = dict(repo=REPO, revision=REVISION, files=FILES,
                  status='verified', elapsed_s=time.monotonic()-start, endpoints=endpoints)
    (args.root/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    emit('finish', **record)

if __name__ == '__main__':
    main()
