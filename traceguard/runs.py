"""Timestamped, never-overwritten result files under results/runs/."""
import json, datetime
from pathlib import Path


def save(name, obj, root="results/runs"):
    Path(root).mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    p, i = Path(root) / f"{stamp}_{name}.json", 1
    while p.exists():
        p = Path(root) / f"{stamp}_{name}_{i}.json"; i += 1
    json.dump(obj, open(p, "w"), indent=1)
    return str(p)
