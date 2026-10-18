"""Verify this handoff's hashes only. Standard library; no model execution."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'artifact_manifest.json').read_text(encoding='utf-8'))
for relative,expected in manifest['files'].items():
    path=(root/relative).resolve()
    assert path.is_relative_to(root.resolve()) and path.is_file(),relative
    assert path.stat().st_size==expected['bytes'],relative
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256'],relative
print(f"PASS: {len(manifest['files'])} files; no inference/training performed")
