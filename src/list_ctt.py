"""List every file in the can-train-and-test Bitbucket repo with its size (manifest of the official release)."""
import json, sys, urllib.request, csv
BASE = "https://api.bitbucket.org/2.0/repositories/brooke-lampe/can-train-and-test/src/master/"
def ls(path):
    url = BASE + path + "?pagelen=100"
    out = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        out += d.get("values", [])
        url = d.get("next")
    return out
rows = []
for s in ["set_01", "set_02", "set_03", "set_04"]:
    for sub in ls(s + "/"):
        if sub["type"] != "commit_directory":
            continue
        for f in ls(sub["path"] + "/"):
            if f["type"] == "commit_file":
                rows.append((s, sub["path"].split("/")[1], f["path"], f.get("size")))
w = csv.writer(sys.stdout)
w.writerow(["set", "subset", "path", "size_bytes"])
for r in rows: w.writerow(r)
