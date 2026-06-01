import json
import sys

path = sys.argv[1]
target = int(sys.argv[2]) if len(sys.argv) > 2 else None
with open(path) as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        if target is not None and i != target:
            continue
        print(json.dumps(json.loads(line), indent=2, ensure_ascii=False))
        print("-" * 80)
