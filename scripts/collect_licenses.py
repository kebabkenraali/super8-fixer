"""Collects the license texts of every Rust crate linked into Super8Fixer.exe into one file.

Usage: python scripts/collect_licenses.py [output]   (default: THIRD_PARTY_LICENSES.txt)
Run from the repository root after `cargo build`, so that the crate sources are in the cargo registry.
"""
import collections
import glob
import json
import os
import subprocess
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "THIRD_PARTY_LICENSES.txt"
meta = json.loads(subprocess.run(["cargo", "metadata", "--format-version", "1", "--filter-platform",
                                  "x86_64-pc-windows-msvc"], capture_output=True, text=True, check=True).stdout)
nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
pkgs = {p["id"]: p for p in meta["packages"]}
seen, stack = set(), [meta["resolve"]["root"]]
while stack:
    i = stack.pop()
    if i in seen:
        continue
    seen.add(i)
    for d in nodes[i]["deps"]:
        if any(k["kind"] in (None, "normal") for k in d["dep_kinds"]):
            stack.append(d["pkg"])
seen.discard(meta["resolve"]["root"])

# group crates by identical license text so that each text is printed once
by_text = collections.OrderedDict()
missing = []
for i in sorted(seen, key=lambda x: (pkgs[x]["name"], pkgs[x]["version"])):
    p = pkgs[i]
    d = os.path.dirname(p["manifest_path"])
    files = sorted(f for pat in ("LICENSE*", "LICENCE*", "COPYING*", "NOTICE*", "COPYRIGHT*", "license*",
                                 "fonts/*.txt")
                   for f in glob.glob(os.path.join(d, pat)) if os.path.isfile(f))
    if not files:
        authors = ", ".join(p.get("authors") or []) or "the crate authors"
        missing.append(f"{p['name']} {p['version']} ({p.get('license')}), copyright {authors}, {p.get('repository') or ''}")
        continue
    for f in files:
        text = open(f, encoding="utf-8", errors="replace").read().strip()
        by_text.setdefault(text, []).append(f"{p['name']} {p['version']} ({os.path.basename(f)}, {p.get('license')})")

with open(out, "w", encoding="utf-8", newline="\n") as o:
    o.write("Super8 Fixer - licenses of the Rust crates linked into Super8Fixer.exe\n")
    o.write("=" * 72 + "\n\n")
    o.write(f"{len(seen)} crates. Where a crate offers a choice of licenses (e.g. \"MIT OR Apache-2.0\"),\n"
            "it is used under the permissive license of that choice (MIT or Apache-2.0; never GPL).\n\n")
    for text, users in by_text.items():
        o.write("-" * 72 + "\nUsed by:\n")
        for u in users:
            o.write(f"  {u}\n")
        o.write("\n" + text + "\n\n")
    if missing:
        o.write("-" * 72 + "\nCrates without a license file in their package (license from Cargo.toml):\n")
        for m in missing:
            o.write(f"  {m}\n")
print(f"{out}: {len(seen)} crates, {len(by_text)} distinct texts, {len(missing)} without a file")
