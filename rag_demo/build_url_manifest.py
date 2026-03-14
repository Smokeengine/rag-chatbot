import os, re, json
from pathlib import Path

ROOT = Path(__file__).parent
data_dir = ROOT.parent / "data"
visited_file = ROOT.parent / "visited.txt"
out_dir = ROOT / "meta"
out_dir.mkdir(parents=True, exist_ok=True)

def url_to_filename(url: str) -> str:
    path = url.replace("https://tilt.colostate.edu/", "")
    path = re.sub(r"[#?].*", "", path)
    path = path.strip("/")
    name = path.replace("/", "_").replace("-", "_")
    if not name:
        name = "index"
    return name + ".txt"

with open(visited_file, "r", encoding="utf-8") as f:
    urls = [ln.strip() for ln in f if ln.startswith("https://")]

manifest = []
for url in urls:
    fname = url_to_filename(url)
    if (data_dir / fname).exists():
        manifest.append({"file": fname, "url": url})

with open(out_dir / "url_manifest.json", "w", encoding="utf-8") as out:
    json.dump(manifest, out, indent=2)

print(f"Created manifest for {len(manifest)} files")
