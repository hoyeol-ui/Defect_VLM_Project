"""Portable dependency injection for unchanged historical workers.

Importing this module never imports torch/Ultralytics or opens private inputs.
Synthetic callbacks must be supplied explicitly; no production fallback on failure.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
from types import FunctionType

from ..entropy import entropy_acquisition, stable_sha256
from ..entropy_extension import Extension
from ..entropy_official import verify_checkpoint_selection
from ..entropy_official_training import run_cell, verify_environment as verify_worker_environment
from . import PortabilityError
from .paths import PathMap, contained, relative_parts
from .protocol import REQUIRED_SOURCES
from .storage import canonical_hash, checked, sha256, write_new


def bind(function, **dependencies):
    """Private globals per invocation; never patch the historical module globals."""
    return FunctionType(function.__code__, {**function.__globals__, **dependencies},
                        function.__name__, function.__defaults__, function.__closure__)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def existing_or_new(path, value):
    if path.exists():
        if read(path) != value:
            raise PortabilityError("portable input/config drift")
    else:
        write_new(path, value)


def normalized_training_args(args, expected_output):
    """Validate Ultralytics' derived output metadata before exact config audit.

    The historical worker checks trainer.save_dir independently of its sealed
    train kwargs. Ultralytics also serializes that derived path in args.yaml.
    No scientific key or unknown extra key is removed here.
    """
    config = {**args, "device": str(args["device"])}
    if "save_dir" in config:
        if (not isinstance(config["save_dir"], str)
                or Path(config["save_dir"]).resolve() != Path(expected_output).resolve()):
            raise PortabilityError("serialized save_dir differs from approved output")
        del config["save_dir"]
    return config


@dataclass(frozen=True)
class JobContext:
    """Method-independent dispatch envelope; plan authorization remains separate."""
    namespace: str
    node: dict
    candidate: dict
    authorization_path: Path
    authorization_sha256: str


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


class WorkerAdapter:
    def __init__(self, *, trainer=None, scorer=None, checkpoint_reader=None):
        self.trainer = trainer
        self.scorer = scorer or entropy_acquisition
        self.checkpoint_reader = checkpoint_reader or load_checkpoint

    def prepare(self, study, layout, registry, context):
        production = ((context.node["kind"] == "training" and self.trainer is None)
                      or (context.node["kind"] == "acquisition" and self.scorer is entropy_acquisition))
        if production:
            # Production imports must come from the source bytes that were approved,
            # not merely a second directory containing a matching-looking manifest.
            root = Path(__file__).resolve().parents[3]
            for rel in REQUIRED_SOURCES:
                checked(root / rel, context.candidate["source_hashes"][rel])
            import platform
            if context.candidate["environment"]["os"] != platform.platform():
                raise PortabilityError("actual OS differs from approved environment")
        bridge = EntropyBridge(study, layout, registry, context, self)
        if production:
            verify_worker_environment(bridge.protocol)
        return bridge

    def run(self, bridge):
        node = bridge.context.node
        if node["kind"] == "acquisition":
            # The same scorer/ranking/pool validation and sealing code as historical.
            def score(checkpoint, records, size, config, **kwargs):
                config = deepcopy(config)
                config["source_sha256"] = {"/".join(relative_parts(k)): v
                                          for k, v in config["source_sha256"].items()}
                return self.scorer(checkpoint, records, size, config, **kwargs)
            bind(Extension._acquire, entropy_acquisition=score)(bridge, node["node_id"])
        elif node["kind"] == "training":
            spec = bridge.prepare_training(node["node_id"])
            if self.trainer is None:
                # Only the Extension construction is injected. Original YOLO call,
                # GPU/package checks, args audit and val-only evaluation are retained.
                bind(run_cell, Extension=lambda root, mode: bridge)(
                    bridge.root, bridge.mode, node["node_id"], bridge.context.authorization_path)
            else:
                artifact = self.trainer(deepcopy(spec), bridge.training_output())
                bridge._complete(node["node_id"], artifact)
        else:
            raise PortabilityError("unsupported adapter job kind")
        bridge.recheck_inputs()
        result = read(bridge.artifact)
        result.update(status="COMPLETE", portable_candidate_sha256=bridge.freeze_sha,
                      authorization_sha256=bridge.context.authorization_sha256,
                      namespace=bridge.mode, acquisition_seed=int(bridge.seed_row["acquisition_seed"]))
        if node["kind"] == "training":
            result["prepared_spec_sha256"] = sha256(bridge.spec_path(node))
            required = {str(Path(result["best_path"])), str(Path(result["last_path"])),
                        str(Path(result["best_path"]).parent.parent / "args.yaml")}
            if not required <= {e["path"] for e in result["files"]}:
                raise PortabilityError("best/last/args file seals required")
        result["files"].append({"path": str(bridge.artifact), "sha256": sha256(bridge.artifact)})
        target = bridge.artifact.with_name("portable_result.json")
        write_new(target, result)
        bridge.audit_result(node, result)
        return target


class EntropyBridge:
    """Entropy policy binding; the shared dispatch/registry schema has no method ID."""
    def __init__(self, study, layout, registry, context, adapter):
        self.study, self.layout, self.registry = study, layout, registry
        self.context, self.adapter = context, adapter
        self.mode = context.namespace
        if self.mode != "entropy":
            raise PortabilityError("no reviewed worker policy registered for this namespace")
        self.root = layout.output(self.mode, "runtime").parent
        self.freeze_sha = canonical_hash(context.candidate)
        self.protocol = deepcopy(study.protocol)
        n = context.node
        rows = [s for s in self.protocol["seeds"] if s["dataset"] == n["dataset"]
                and int(s["acquisition_index"]) == n["acquisition_index"]]
        if len(rows) != 1 or n["trajectory"] != f"{n['dataset']}_a{n['acquisition_index']:02d}":
            raise PortabilityError("dataset/acquisition identity mismatch")
        self.seed_row = rows[0]
        if n["kind"] == "training" and (n["slot"] not in ("A", "B")
                or n["training_seed"] != int(rows[0][f"training_seed_{n['slot'].lower()}"])
                or n["acquisition_seed"] != int(rows[0]["acquisition_seed"])):
            raise PortabilityError("learner/acquisition seed mismatch")
        env = context.candidate["environment"]
        self.protocol["environment"].update(gpu=env["gpu_model"],
            python_executable=sys.executable, ultralytics_sources=env["ultralytics_sources"])
        self.input_seals = []
        self.data_cache = {}
        self.r0 = {}
        self.artifact = None
        self.mapper = PathMap(context.candidate["path_map"], "windows" if os.name == "nt" else "linux")
        self.initial = self.asset("pretrained")
        if sha256(self.initial) != study.protocol["pretrained"]["sha256"]:
            raise PortabilityError("wrong initial weight")
        self.protocol["pretrained"]["path"] = str(self.initial)
        # Only references needed by the authorized trajectory are resolved.
        tid = context.node["trajectory"]
        ref_id = f"{tid}_SHARED_A_R0"
        ref = deepcopy(study.r0[ref_id])
        for role, key in (("checkpoints", "best"), ("manifests", "manifest")):
            path = self.asset(role, ref_id)
            checked(path, ref[f"{key}_sha256"])
            ref[f"{key}_path"] = str(path)
        self.r0[ref_id] = ref
        self.dataset(context.node["dataset"])
        self.check_dependencies()

    def spec_path(self, node):
        return self.layout.output(self.mode, f"training_inputs/{self.mode}/{node['depends_on'][0]}/{node['slot']}_spec.json")

    def remember(self, path, digest):
        path = checked(path, digest)
        self.input_seals.append((path, digest))
        return path

    def asset(self, role, reference=None):
        choices = [a for a in self.context.candidate["assets"] if a["role"] == role
                   and (reference is None or a.get("reference") == reference)]
        if len(choices) != 1:
            raise PortabilityError(f"exactly one sealed asset required: {role}/{reference}")
        a = choices[0]
        return self.remember(self.layout.input(role, a["relative"]), a["sha256"])

    def node(self, node_id):
        matches = [n for n in self.context.candidate["plan"] if n["node_id"] == node_id]
        if len(matches) != 1:
            raise PortabilityError("node outside authorized plan")
        return matches[0]

    def dataset(self, name):
        if name != self.context.node["dataset"] or name not in self.study.dataset_hashes:
            raise PortabilityError("dataset ID mismatch")
        if name in self.data_cache:
            return self.data_cache[name]
        path = self.asset("manifests", f"dataset:{name}")
        checked(path, self.study.dataset_hashes[name])
        data = read(path)
        core = {"train_ids", "validation_ids", "images", "labels", "names"}
        metadata = {"dataset", "locked_test_images_indexed"}
        if set(data) not in (core, core | metadata):
            raise PortabilityError("dataset schema/split mismatch; test endpoint forbidden")
        if metadata <= set(data) and (data["dataset"] != name
                or type(data["locked_test_images_indexed"]) is not int
                or data["locked_test_images_indexed"] != 0):
            raise PortabilityError("unapproved dataset metadata or locked-test content")
        # Descriptive metadata never becomes a worker input or endpoint.
        data = {key: data[key] for key in core}
        train, val = data["train_ids"], data["validation_ids"]
        if (not train or not val or len(set(train)) != len(train) or len(set(val)) != len(val)
                or set(train) & set(val) or set(data["images"]) != set(train) | set(val)
                or set(data["labels"]) != set(train) | set(val)):
            raise PortabilityError("dataset split mismatch")
        if len(train) != self.protocol["budgets"][name]["acquisition_universe_n"]:
            raise PortabilityError("acquisition split budget mismatch")
        if any(len(i) != 64 or any(c not in "0123456789abcdef" for c in i) for i in train + val):
            raise PortabilityError("canonical image ID mismatch")
        for section, role in (("images", "dataset"), ("labels", "labels")):
            for image_id, entry in data[section].items():
                # Hash-anchored index membership is the allowlist, not filename heuristics.
                if set(entry) != {"path", "sha256"}:
                    raise PortabilityError("unapproved image/label endpoint metadata")
                target = self.mapper.translate(entry["path"])
                native = contained(self.layout.inputs[role], str(target))
                entry["path"] = str(native)
                # Acquisition never opens labels; training's copy verifies them later.
                if role == "dataset" or self.context.node["kind"] == "training":
                    self.remember(native, entry["sha256"])
        self.data_cache[name] = data
        return data

    def completed(self, node_id):
        result = self.registry.completed(node_id)
        if result.get("portable_candidate_sha256") != self.freeze_sha or result.get("namespace") != self.mode:
            raise PortabilityError("foreign result/freeze/namespace")
        return result

    def state(self, node_id):
        return self.registry.state(node_id)

    def _claim(self, node_id, expected):
        if node_id != self.context.node["node_id"] or self.state(node_id)["state"] != "RUNNING":
            raise PortabilityError("worker has no authorized registry claim")
        return self.state(node_id)["attempt"]

    def _complete(self, node_id, artifact):
        self._claim(node_id, None)
        self.artifact = contained(self.root, artifact)

    def fail(self, node_id):
        # The outer execute transaction records FAILED, exactly once.
        self._claim(node_id, None)

    def verify_authorization(self, authorization, node_id):
        if Path(authorization) != self.context.authorization_path or node_id != self.context.node["node_id"]:
            raise PortabilityError("worker authorization scope mismatch")
        return read(self.remember(authorization, self.context.authorization_sha256))

    policy_checkpoint = Extension.policy_checkpoint
    prior_labeled = Extension.prior_labeled
    _copy_sealed = Extension._copy_sealed

    def check_dependencies(self):
        n = self.context.node
        for r in range(1, n["round"] + 1):
            for suffix in (f"Q_R{r}", f"Entropy_A_R{r}", f"Entropy_B_R{r}"):
                self.node(f"{n['trajectory']}_{suffix}")
        if n["round"] > 1:
            self.round_gate(n["trajectory"], n["round"] - 1)
        if n["kind"] == "acquisition":
            expected = (f"{n['trajectory']}_SHARED_A_R0" if n["round"] == 1 else
                        f"{n['trajectory']}_Entropy_A_R{n['round'] - 1}")
            if n["policy_slot"] != "A" or n["depends_on"] != [expected]:
                raise PortabilityError("Entropy parent must be previous-round A")
            self.policy_checkpoint(n["trajectory"], n["round"])
        else:
            self.audit_query(self.node(n["depends_on"][0]), self.completed(n["depends_on"][0]))

    def training_output(self):
        n = self.context.node
        return self.layout.output(self.mode, f"outcomes_private/{self.mode}/{n['node_id']}/attempt_01")

    def prepare_training(self, node_id):
        node = self.node(node_id)
        q = self.completed(node["depends_on"][0])
        self.audit_query(self.node(node["depends_on"][0]), q)
        config = deepcopy(self.study.training)
        config["model"] = str(self.initial)
        existing_or_new(self.root / "training_config.json", config)
        spec = Extension.prepare_training(self, node_id)
        actual = {**spec["training_config"], "project": str(self.training_output().parent),
                  "name": self.training_output().name, "exist_ok": False}
        self.study.verify_config(actual, node)
        for key in ("data_yaml", "manifest"):
            self.remember(Path(spec[f"{key}_path"]), spec[f"{key}_sha256"])
        return spec

    def audit_query(self, node, result):
        if any(result.get(k) != value for k, value in node.items()):
            raise PortabilityError("acquisition identity mismatch")
        data = self.dataset(node["dataset"])
        manifest = contained(self.root, result["manifest_path"])
        ids = self.remember(manifest, result["manifest_sha256"]).read_text().splitlines()
        prior = self.prior_labeled(node["trajectory"], node["round"])
        score_path = next(e["path"] for e in result["files"] if Path(e["path"]).name == "scores.json")
        scores = read(score_path)
        selected = scores["selected_ids"]
        if (len(ids) != self.study.protocol["budgets"][node["dataset"]]["cumulative_counts"][node["round"]]
                or len(ids) != len(set(ids)) or len(selected) != node["query_size"]
                or len(selected) != len(set(selected)) or set(prior) & set(selected)
                or set(ids) != set(prior) | set(selected) or not set(ids) <= set(data["train_ids"])
                or set(ids) & set(data["validation_ids"])):
            raise PortabilityError("manifest budget/duplicate/reselection/split mismatch")
        if result["policy_checkpoint"] != self.policy_checkpoint(node["trajectory"], node["round"]):
            raise PortabilityError("query checkpoint parent mismatch")

    def audit_training(self, node, result):
        import yaml
        q = self.completed(node["depends_on"][0])
        spec = read(self.remember(self.spec_path(node), result["prepared_spec_sha256"]))
        if (any(result.get(k) != v for k, v in node.items())
                or result.get("manifest_sha256") != q["manifest_sha256"]
                or result.get("manifest_path") != q["manifest_path"]
                or result.get("initialization_path") != str(self.initial)
                or result.get("initialization_sha256") != sha256(self.initial)
                or result.get("cold_start") is not True or result.get("resume") is not False
                or result.get("epochs_completed") != 100
                or result.get("checkpoint_selection_split") != "validation"
                or result.get("locked_test_accessed") is not False):
            raise PortabilityError("training identity/manifest/cold-start contract mismatch")
        best = self.remember(contained(self.root, result["best_path"]), result["best_sha256"])
        last = self.remember(contained(self.root, result["last_path"]), result["last_sha256"])
        if best == last or best.name != "best.pt" or last.name != "last.pt":
            raise PortabilityError("distinct best/last checkpoints required")
        args = yaml.safe_load((best.parent.parent / "args.yaml").read_text())
        expected_output = self.layout.output(self.mode, f"outcomes_private/{self.mode}/{node['node_id']}/attempt_01")
        self.study.verify_config(normalized_training_args(args, expected_output), node)
        if args["model"] != str(self.initial) or args["data"] != spec["data_yaml_path"]:
            raise PortabilityError("wrong initial weight or dataset YAML")
        if Path(args["project"]) / args["name"] != expected_output or best.parent.parent != expected_output:
            raise PortabilityError("training output mismatch")
        if result.get("training_config_sha256") != stable_sha256(spec["training_config"]):
            raise PortabilityError("training config hash mismatch")
        self.remember(contained(self.root, spec["data_yaml_path"]), spec["data_yaml_sha256"])
        dataset = yaml.safe_load(Path(spec["data_yaml_path"]).read_text())
        if set(dataset) != {"path", "train", "val", "names"}:
            raise PortabilityError("locked-test YAML forbidden")
        source = self.dataset(node["dataset"])
        expected_ids = {"train": Path(q["manifest_path"]).read_text().splitlines(), "val": source["validation_ids"]}
        for key, seal in (("train", "train_list_sha256"), ("val", "validation_list_sha256")):
            listing = self.remember(contained(self.root, dataset[key]), spec[seal])
            paths = [contained(self.root, p) for p in listing.read_text().splitlines()]
            if len(paths) != len(expected_ids[key]) or {p.stem for p in paths} != set(expected_ids[key]):
                raise PortabilityError("prepared split membership mismatch")
            for p in paths:
                self.remember(p, source["images"][p.stem]["sha256"])
                self.remember(p.parent.parent / "labels" / (p.stem + ".txt"), source["labels"][p.stem]["sha256"])
        b, l = self.adapter.checkpoint_reader(best), self.adapter.checkpoint_reader(last)
        verify_checkpoint_selection(b, l)
        for checkpoint in (b, l):
            checkpoint_args = normalized_training_args(checkpoint["train_args"], expected_output)
            for key, value in normalized_training_args(args, expected_output).items():
                actual = checkpoint_args.get(key)
                if (str(actual) != str(value) if key == "device" else actual != value):
                    raise PortabilityError("checkpoint training arguments mismatch")

    def audit_result(self, node, result):
        if node["kind"] == "training":
            self.audit_training(node, result)
        else:
            self.audit_query(node, result)

    def round_gate(self, trajectory, round_idx):
        qid = f"{trajectory}_Q_R{round_idx}"
        q = self.completed(qid)
        self.audit_query(self.node(qid), q)
        cells = []
        for slot in ("A", "B"):
            cid = f"{trajectory}_Entropy_{slot}_R{round_idx}"
            result = self.completed(cid)
            self.audit_training(self.node(cid), result)
            cells.append({k: result[k] for k in ("node_id", "best_sha256", "last_sha256",
                "manifest_sha256", "training_config_sha256", "prepared_spec_sha256")})
        gate = {"status": "PASS", "trajectory": trajectory, "round": round_idx,
                "manifest_sha256": q["manifest_sha256"], "policy_slot": "A",
                "candidate_sha256": self.freeze_sha, "locked_test_accesses": 0, "cells": cells}
        path = self.layout.output(self.mode, f"gates/{trajectory}_R{round_idx}.json")
        existing_or_new(path, gate)
        self.registry.seal_gate(f"{trajectory}_R{round_idx}", path)
        return gate

    def maybe_gate(self):
        n = self.context.node
        if n["kind"] == "training":
            peers = [f"{n['trajectory']}_Entropy_{slot}_R{n['round']}" for slot in ("A", "B")]
            if all(self.registry.state(cid)["state"] == "COMPLETE" for cid in peers):
                self.round_gate(n["trajectory"], n["round"])

    def recheck_inputs(self):
        for path, digest in self.input_seals:
            checked(path, digest)
