"""Download every file of the official can-train-and-test release (Bitbucket brooke-lampe/can-train-and-test, master)
into data/raw/ctt/<set>/<subset>/<file>, then write the SHA-256 of each file to results/ctt_download_sha256.csv.

The file list is results/ctt_repo_listing.csv (written by list_ctt.py). Needs curl.
"""
import concurrent.futures as cf
import csv
import hashlib
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data/raw/ctt")
RAW = "https://bitbucket.org/brooke-lampe/can-train-and-test/raw/master/"


def get(r):
    dst = os.path.join(DEST, r["path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    want = int(r["size_bytes"])
    for _ in range(4):
        if os.path.exists(dst) and os.path.getsize(dst) == want:
            break
        subprocess.run(["curl", "-s", "-S", "-L", "--retry", "3", "-m", "1800", "-o", dst, RAW + r["path"]],
                       check=False)
    ok = os.path.exists(dst) and os.path.getsize(dst) == want
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest() if ok else ""
    return r["path"], want, ok, h


if __name__ == "__main__":
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results/ctt_repo_listing.csv"))))
    out = os.path.join(ROOT, "results/ctt_download_sha256.csv")
    with cf.ThreadPoolExecutor(4) as ex, open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "size_bytes", "ok", "sha256"])
        for res in ex.map(get, rows):
            w.writerow(res)
            f.flush()
    print("done")
