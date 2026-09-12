"""Inert historical snapshots and separate, append-only attempt records."""
from contextlib import closing, contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from . import PortabilityError
from .paths import contained, disjoint


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path, expected):
    if sha256(path) != expected:
        raise PortabilityError(f"SHA-256 mismatch: {path}")
    return Path(path)


def write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def canonical_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@contextmanager
def historical_snapshot(source, expected_sha256, scratch_root):
    source, scratch_root = Path(source).resolve(), Path(scratch_root).resolve()
    if not source.is_file() or not disjoint(source.parent, scratch_root):
        raise PortabilityError("snapshot must be outside historical directory")

    def inventory():
        state = []
        for p in (source, Path(str(source) + "-wal"), Path(str(source) + "-shm"), Path(str(source) + "-journal")):
            if p.exists():
                stat = p.stat()
                if p != source and str(p).endswith(("-wal", "-journal")) and stat.st_size:
                    raise PortabilityError("pending SQLite journal: obtain a quiescent archival snapshot first")
                state.append((str(p), stat.st_size, stat.st_mtime_ns, sha256(p)))
            else:
                state.append((str(p), None))
        return state

    before = inventory()
    checked(source, expected_sha256)
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="registry-audit-", dir=scratch_root) as directory:
        snapshot = Path(directory) / "registry.sqlite"
        shutil.copyfile(source, snapshot)
        checked(snapshot, expected_sha256)
        if inventory() != before:
            raise PortabilityError("source changed while copying SQLite snapshot")
        connection = sqlite3.connect(snapshot.as_uri() + "?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            yield connection
        finally:
            connection.close()
            if inventory() != before:
                raise PortabilityError("historical SQLite or sidecar changed during audit")


class Registry:
    def __init__(self, layout, namespace):
        self.layout, self.namespace = layout, namespace
        self.path = layout.registry(namespace)

    def _connect(self):
        if self.path != self.layout.registry(self.namespace):
            raise PortabilityError("registry path changed")
        if self.path.exists() and self.path.stat().st_nlink != 1:
            raise PortabilityError("registry must not be a hard link to another file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA synchronous=FULL")
        return c

    def initialize(self, plan, release_hash):
        if not plan or len({p["node_id"] for p in plan}) != len(plan):
            raise PortabilityError("invalid registry plan")
        with closing(self._connect()) as c:
            with c:
                c.execute("BEGIN IMMEDIATE")
                c.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                c.execute("CREATE TABLE IF NOT EXISTS jobs (node_id TEXT PRIMARY KEY, state TEXT NOT NULL, attempt INTEGER NOT NULL, spec TEXT NOT NULL, artifact TEXT, sha256 TEXT)")
                c.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, node_id TEXT, state TEXT, detail TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP)")
                expected = {"release": release_hash, "plan": canonical_hash(plan), "namespace": self.namespace}
                stored = dict(c.execute("SELECT key,value FROM metadata"))
                if stored:
                    if stored != expected:
                        raise PortabilityError("registry belongs to a different release or namespace")
                else:
                    c.executemany("INSERT INTO metadata VALUES (?,?)", expected.items())
                    c.executemany("INSERT INTO jobs VALUES (?, 'PENDING',0,?,NULL,NULL)",
                                  [(p["node_id"], json.dumps(p, sort_keys=True)) for p in plan])
                actual = [json.loads(r[0]) for r in c.execute("SELECT spec FROM jobs ORDER BY node_id")]
                if actual != sorted(plan, key=lambda p: p["node_id"]):
                    raise PortabilityError("registry plan was modified")

    def state(self, node):
        c = self._connect()
        try:
            row = c.execute("SELECT * FROM jobs WHERE node_id=?", (node,)).fetchone()
            if row is None:
                raise PortabilityError("unknown node")
            return dict(row)
        finally:
            c.close()

    def transition(self, node, expected, new, *, artifact=None, detail=""):
        if (expected, new) not in {("PENDING", "RUNNING"), ("RUNNING", "COMPLETE"), ("RUNNING", "FAILED")}:
            raise PortabilityError("retry/resume requires separate explicit reconciliation")
        digest = None
        if new == "COMPLETE":
            artifact = contained(self.layout.output_root / self.namespace, artifact)
            if artifact.suffix != ".json" or not artifact.is_file():
                raise PortabilityError("sealed result JSON required")
            result = json.loads(artifact.read_text(encoding="utf-8"))
            if result.get("node_id") != node or result.get("status") != "COMPLETE" or not result.get("files"):
                raise PortabilityError("result identity or file seal missing")
            for entry in result["files"]:
                checked(contained(artifact.parent, entry["path"]), entry["sha256"])
            digest = sha256(artifact)
        c = self._connect()
        try:
            with c:
                cursor = c.execute("UPDATE jobs SET state=?,attempt=attempt+?,artifact=?,sha256=? WHERE node_id=? AND state=?",
                                   (new, int(new == "RUNNING"), str(artifact) if artifact else None, digest, node, expected))
                if cursor.rowcount != 1:
                    raise PortabilityError("duplicate claim or invalid job transition")
                c.execute("INSERT INTO events(node_id,state,detail) VALUES (?,?,?)", (node, new, detail))
        finally:
            c.close()

    def completed(self, node):
        row = self.state(node)
        if row["state"] != "COMPLETE":
            raise PortabilityError("dependency is not complete")
        path = contained(self.layout.output_root / self.namespace, row["artifact"])
        result = json.loads(checked(path, row["sha256"]).read_text(encoding="utf-8"))
        for entry in result["files"]:
            checked(contained(path.parent, entry["path"]), entry["sha256"])
        return result

    def seal_gate(self, gate_id, path):
        path = contained(self.layout.output_root / self.namespace, path)
        digest = sha256(path)
        with closing(self._connect()) as c, c:
            c.execute("CREATE TABLE IF NOT EXISTS gates (gate_id TEXT PRIMARY KEY, path TEXT, sha256 TEXT)")
            row = c.execute("SELECT path,sha256 FROM gates WHERE gate_id=?", (gate_id,)).fetchone()
            if row and tuple(row) != (str(path), digest):
                raise PortabilityError("round gate seal drift")
            if not row:
                c.execute("INSERT INTO gates VALUES (?,?,?)", (gate_id, str(path), digest))
                c.execute("INSERT INTO events(node_id,state,detail) VALUES (?, 'ROUND_GATE_PASS', ?)", (gate_id, digest))
