from pathlib import Path
import yaml

def load_yaml(path):
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).with_name(p.name)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
