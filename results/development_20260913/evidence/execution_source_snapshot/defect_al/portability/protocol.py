"""A separately authorized environment overlay over the unchanged study seals."""
import json
from pathlib import Path, PureWindowsPath
import re

from . import PortabilityError
from .paths import PathMap, relative_parts
from .storage import canonical_hash, checked, sha256

ENGINE_SHA = "0d3deb6656b9b22ddbb8d63f17ef6721adac4f3120560528c44954f8713d00fb"
OFFICIAL_SHA = "e40842787a93d2e06304f8da7eea5ea93f2125c7bd15982bbce6610903505e6a"
ENV_FIELDS = {"os", "gpu_model", "python", "cuda", "pytorch", "ultralytics", "packages", "ultralytics_sources"}
REQUIRED_SOURCES = {f"src/defect_al/portability/{name}.py" for name in
                    ("__init__", "paths", "host", "storage", "protocol", "execution", "adapter")}
REQUIRED_SOURCES |= {f"src/defect_al/{name}.py" for name in
                     ("entropy", "entropy_extension", "entropy_official", "entropy_official_training")}


class Study:
    """Read only the public protocol files needed to bind portable configuration."""
    def __init__(self, repository):
        root = Path(repository) / "entropy_extension_v1"
        engine = json.loads(checked(root / "freeze.json", ENGINE_SHA).read_text(encoding="utf-8"))
        checked(root / "official_release_v1/freeze.json", OFFICIAL_SHA)
        hashes = {PureWindowsPath(e["path"]).name: e["sha256"] for e in engine["files"]}

        def read(name):
            return json.loads(checked(root / name, hashes[name]).read_text(encoding="utf-8"))

        self.protocol = read("protocol.json")
        self.training = read("training_config.json")
        self.cells = read("planned_cells.json")
        self.queries = read("planned_acquisitions.json")
        self.r0 = read("historical_r0.json")
        self.dataset_hashes = {ds: hashes[f"{ds}.json"] for ds in ("GC10-DET", "NEU-DET")}
        self.hashes = {name: hashes[name] for name in ("protocol.json", "training_config.json", "planned_cells.json", "planned_acquisitions.json", "historical_r0.json")}

    def training_config(self, *, model, data, seed, output, node):
        if node not in self.cells or seed != node["training_seed"]:
            raise PortabilityError("training seed or cell is not in the sealed study")
        config = dict(self.training)
        config.update(model=str(model), data=str(data), seed=seed,
                      project=str(Path(output).parent), name=Path(output).name)
        return config

    def verify_config(self, config, node):
        # Every original knob is checked, not only epochs/batch/optimizer.
        expected = self.training_config(model=config["model"], data=config["data"],
                                        seed=config["seed"], output=Path(config["project"]) / config["name"], node=node)
        if config != expected:
            raise PortabilityError("scientific training configuration changed")

    def verify_plan(self, namespace, plan):
        if not plan or len({n["node_id"] for n in plan}) != len(plan):
            raise PortabilityError("nonempty unique plan required")
        if namespace == "entropy":
            lookup = {n["node_id"]: n for n in self.cells + self.queries}
        elif namespace == "smoke":
            lookup = {"input_check": {"node_id": "input_check", "kind": "input_check", "depends_on": []}}
        else:
            raise PortabilityError("namespace reserved; separate execution plan integration required")
        if any(n != lookup.get(n["node_id"]) for n in plan):
            raise PortabilityError("plan changed a seed, budget, role or dependency")


def verify_environment(study, observed, expected=None):
    if set(observed) != ENV_FIELDS or any(not observed[k] for k in ENV_FIELDS):
        raise PortabilityError("complete observed environment required")
    original = study.protocol["environment"]
    checks = {"python": original["python_version"], "cuda": original["cuda"],
              "pytorch": original["packages"]["torch"], "ultralytics": original["packages"]["ultralytics"]}
    if any(observed[k] != v for k, v in checks.items()) or observed["packages"] != original["packages"]:
        raise PortabilityError("package/runtime drift requires a separate reviewed protocol")
    source_keys = {"/".join(relative_parts(k)): v for k, v in original["ultralytics_sources"].items()}
    if observed["ultralytics_sources"] != source_keys:
        raise PortabilityError("installed Ultralytics source hashes differ")
    if expected is not None and observed != expected:
        raise PortabilityError("actual host differs from authorized environment")


def build_candidate(study, layout, namespace, plan, environment, path_map, assets, *, commit, sources):
    """Build a candidate only. This function never grants execution authorization."""
    layout.validate()
    study.verify_plan(namespace, plan)
    verify_environment(study, environment)
    for target in ("windows", "linux"):
        PathMap(path_map, target)
    if (not re.fullmatch(r"[0-9a-f]{40}", commit) or not REQUIRED_SOURCES <= set(sources)
            or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in sources.values())):
        raise PortabilityError("source commit and all portability guard hashes required")
    if not assets or any(a.get("split") not in ("acquisition", "development", "metadata") for a in assets):
        raise PortabilityError("assets must explicitly exclude locked-test")
    for asset in assets:
        role = asset["role"]
        file = layout.input(role, asset["relative"])
        checked(file, asset["sha256"])
        if role == "pretrained" and asset["sha256"] != study.protocol["pretrained"]["sha256"]:
            raise PortabilityError("pretrained YOLO weight changed")
    if {a["role"] for a in assets} != set(layout.inputs):
        raise PortabilityError("asset list must cover every configured input role")
    return {"schema": 1, "status": "CANDIDATE", "parent_engine_sha256": ENGINE_SHA,
            "parent_official_sha256": OFFICIAL_SHA, "study_hashes": study.hashes,
            "namespace": namespace, "environment": environment, "source_commit": commit,
            "source_hashes": sources, "path_map": path_map, "assets": assets,
            "input_roots": {k: str(v) for k, v in layout.inputs.items()},
            "historical_roots": [str(p) for p in layout.historical_roots],
            "output_root": str(layout.output_root), "registry": str(layout.registry(namespace)),
            "plan": plan, "config_hash": canonical_hash({"training": study.training, "inference": study.protocol["inference"]}),
            "pretrained_sha256": study.protocol["pretrained"]["sha256"],
            "locked_test_endpoint": None, "locked_test_access": False,
            "authorization": None, "resume": False, "policy_learner": "A"}


def verify_authorization(candidate_path, authorization_path, expected_authorization_sha256,
                         study, layout, environment, node_id, source_root, source_commit):
    """The external authorization seal binds the candidate bytes and exact scope."""
    candidate_path = Path(candidate_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    auth = json.loads(checked(authorization_path, expected_authorization_sha256).read_text(encoding="utf-8"))
    required = {"status", "candidate_sha256", "namespace", "allowed_nodes", "locked_test_access", "environment_change_approved", "resume"}
    if set(auth) != required or auth["status"] != "AUTHORIZED" or auth["candidate_sha256"] != sha256(candidate_path):
        raise PortabilityError("authorization/candidate seal mismatch")
    if auth["locked_test_access"] is not False or auth["resume"] is not False or auth["environment_change_approved"] is not True:
        raise PortabilityError("environment change must be explicitly approved; test/resume forbidden")
    if candidate["locked_test_endpoint"] is not None or candidate["locked_test_access"] is not False:
        raise PortabilityError("locked-test endpoint disabled")
    # Rebuild every bound property to catch layout/config/parent/role manipulation.
    rebuilt = build_candidate(study, layout, candidate["namespace"], candidate["plan"],
                              candidate["environment"], candidate["path_map"], candidate["assets"],
                              commit=source_commit, sources=candidate["source_hashes"])
    if candidate != rebuilt or candidate["source_commit"] != source_commit:
        raise PortabilityError("candidate does not match active protocol/layout/source")
    verify_environment(study, environment, candidate["environment"])
    for rel, expected in candidate["source_hashes"].items():
        from .paths import contained
        checked(contained(source_root, Path(source_root).joinpath(*relative_parts(rel))), expected)
    ids = [n["node_id"] for n in candidate["plan"]]
    # The candidate retains the dependency plan; execution can be authorized
    # for a strictly smaller subset (e.g. one explicitly approved recovery cell).
    allowed = auth["allowed_nodes"]
    if (auth["namespace"] != candidate["namespace"] or not isinstance(allowed, list)
            or not allowed or any(not isinstance(n, str) for n in allowed)
            or len(set(allowed)) != len(allowed) or not set(allowed) <= set(ids)
            or node_id not in allowed):
        raise PortabilityError("job outside exact authorized scope")
    return candidate


def deny_locked_test(*args, **kwargs):
    raise PortabilityError("locked-test endpoint is unconditionally disabled")
