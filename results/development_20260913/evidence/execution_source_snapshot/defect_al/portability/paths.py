"""Explicit path translation without rewriting sealed historical documents."""
from pathlib import Path, PurePosixPath, PureWindowsPath

from . import PortabilityError

ROLES = {"dataset", "labels", "pretrained", "features", "manifests", "checkpoints", "historical_registry"}
NAMESPACES = {"smoke", "comparison", "entropy", "full_data", "followup"}


def relative_parts(value):
    # Backslashes in historical JSON mean separators even on Linux.
    value = str(value)
    win = PureWindowsPath(value)
    parts = tuple(value.replace("\\", "/").split("/"))
    if not parts or win.drive or win.root or value.startswith("/") or any(
        p in ("", "..", ".") or ":" in p or "\x00" in p for p in parts
    ):
        raise PortabilityError("unsafe relative path")
    return parts


def contained(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path != root and root not in path.parents:
        raise PortabilityError("path escapes configured root")
    return path


def disjoint(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a != b and a not in b.parents and b not in a.parents


class PathMap:
    def __init__(self, entries, target_os):
        if target_os not in ("windows", "linux") or not entries:
            raise PortabilityError("explicit path map and target OS required")
        self.entries, self.target_os = entries, target_os
        for entry in entries:
            for key, path_type in (("windows", PureWindowsPath), ("linux", PurePosixPath)):
                target = path_type(entry[key])
                if not target.is_absolute() or ".." in target.parts:
                    raise PortabilityError("mapping target must be absolute without traversal")
            source = str(entry["source"])
            source_type = PureWindowsPath if PureWindowsPath(source).drive else PurePosixPath
            if not source_type(source).is_absolute() or ".." in source_type(source).parts:
                raise PortabilityError("mapping source must be absolute without traversal")

    def translate(self, source):
        windows = bool(PureWindowsPath(str(source)).drive)
        source_type = PureWindowsPath if windows else PurePosixPath
        source = source_type(source)
        if not source.is_absolute() or ".." in source.parts:
            raise PortabilityError("source must be an absolute canonical path")
        matches = []
        for entry in self.entries:
            prefix = source_type(entry["source"])
            try:
                tail = source.relative_to(prefix)
            except ValueError:
                continue
            target_type = PureWindowsPath if self.target_os == "windows" else PurePosixPath
            target = target_type(entry[self.target_os])
            if not target.is_absolute():
                raise PortabilityError("mapping target must be absolute")
            matches.append((len(prefix.parts), target.joinpath(*tail.parts)))
        if not matches:
            raise PortabilityError("unmapped historical path")
        matches.sort(key=lambda item: item[0], reverse=True)
        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            raise PortabilityError("ambiguous path mapping")
        return matches[0][1]


class Layout:
    def __init__(self, inputs, output_root, historical_roots):
        if set(inputs) != ROLES or not historical_roots:
            raise PortabilityError("all input roles and historical roots are required")
        all_paths = list(inputs.values()) + [output_root] + list(historical_roots)
        if not all(Path(p).is_absolute() for p in all_paths):
            raise PortabilityError("native absolute paths required")
        self.inputs = {k: Path(v).resolve() for k, v in inputs.items()}
        self.output_root = Path(output_root).resolve()
        self.historical_roots = [Path(p).resolve() for p in historical_roots]
        self.protected = self.historical_roots + list(self.inputs.values())
        self.validate()

    def validate(self):
        if any(not disjoint(self.output_root, p) for p in self.protected):
            raise PortabilityError("output overlaps historical or input storage")

    def input(self, role, relative):
        if role not in ROLES:
            raise PortabilityError("unknown input role")
        return contained(self.inputs[role], self.inputs[role].joinpath(*relative_parts(relative)))

    def output(self, namespace, relative):
        self.validate()
        if namespace not in NAMESPACES:
            raise PortabilityError("unknown result namespace")
        base = contained(self.output_root, self.output_root / namespace)
        target = contained(base, base.joinpath(*relative_parts(relative)))
        if any(not disjoint(target, p) for p in self.protected):
            raise PortabilityError("output aliases protected input")
        return target

    def registry(self, namespace):
        return self.output(namespace, "registry.sqlite")
