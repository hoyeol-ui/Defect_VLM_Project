import csv
import itertools
import json
import random
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

try:
    from bert_score import score as bertscore_score
except ImportError:
    bertscore_score = None

# =============================================================================
# [1] Experiment Config
# =============================================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SBERT_ID = "sentence-transformers/all-MiniLM-L6-v2"

EXPERIMENT_NAME = "consistency_v3_multidataset_sbert_first"

SAMPLES_PER_CLASS = 10
SEED = 42

DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 64

LOW_K = 10
HIGH_K = 10
RANDOM_K = 10

USE_BERTSCORE_FULL = False
USE_BERTSCORE_CANDIDATES = False

# 추가된 다중 데이터셋 경로 설정 (호열님 로컬 경로 맞춤)
DATASET_CONFIGS = {
    "MVTec_MetalNut": {
        "path": Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project/data/MVTec/archive/metal_nut"),
        "nested": True  # 클래스 하위 폴더(train/test/good/defect 등)가 있는 경우
    },
    "Kolektor": {
        "path": Path(
            "/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project/data/Kolektor/kolektorsdd2-DatasetNinja/train/img"),
        "nested": False  # 이미지가 바로 폴더에 있는 경우
    },
    "NEU-DET": {
        "path": Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project/data/NEU-DET/train/images"),
        "nested": True
    }
}

REPHRASE_PROMPTS = [
    "Describe the visible defect on this metal surface in one short sentence.",
    "In one short sentence, describe the defect visible on this metal surface.",
    "Briefly describe the visible defect on the metal surface in one sentence.",
    "Provide a one-sentence description of the visible defect on this metal surface.",
    "State in one short sentence what defect is visible on this metal surface.",
]

VIEWPOINT_PROMPTS = [
    "Describe the defect by focusing on its location and affected area in one short sentence.",
    "Describe the defect by focusing on its shape or structural pattern in one short sentence.",
    "Describe the defect by focusing on its texture, surface condition, or appearance in one short sentence.",
    "Describe the defect by focusing on whether the damaged region is localized or spread out in one short sentence.",
    "Describe the defect by focusing on the overall visual impression of the damaged surface in one short sentence.",
]


@dataclass
class GenerationConfig:
    device: str
    model_id: str
    sbert_id: str
    seed: int
    do_sample: bool
    temperature: float
    top_p: float
    max_new_tokens: int


METHOD_UPSTREAM_SOURCES = {
    "selfcheckgpt": {
        "official_repo": "https://github.com/potsawee/selfcheckgpt",
        "adaptation_note": "SelfCheckGPT uses multiple generations to estimate self-consistency."
    },
    "semantic_uncertainty": {
        "official_repo": "https://github.com/lorenzkuhn/semantic_uncertainty",
        "adaptation_note": "Semantic uncertainty focuses on meaning-level uncertainty rather than surface-form variation."
    },
    "badge": {
        "official_repo": "https://github.com/JordanAsh/badge",
        "adaptation_note": "BADGE is cited as active-learning method lineage."
    },
    "coreset": {
        "official_repo": "https://github.com/ozansener/active_learning_coreset",
        "adaptation_note": "Core-Set is cited as diversity-based active-learning lineage."
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_project_root() -> Path:
    # 파이참 프로젝트 루트 설정 (실행 위치 기준)
    return Path(__file__).resolve().parent


# 데이터셋별로 폴더 분리를 위한 수정
def prepare_output_dirs(dataset_name: str) -> Dict[str, Path]:
    root = get_project_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = root / "logs" / EXPERIMENT_NAME / f"{timestamp}_{dataset_name}"

    dirs = {
        "base": base_dir,
        "jsonl": base_dir / "jsonl",
        "summary": base_dir / "summary",
        "plots": base_dir / "plots",
        "manifest": base_dir / "manifest",
        "candidates": base_dir / "candidates",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


# 데이터셋 구조(Nested 유무)에 맞춰 동작하도록 통합
def sample_images_universal(config_name: str, samples_per_class: int) -> List[Path]:
    cfg = DATASET_CONFIGS[config_name]
    root_path = cfg["path"]

    if not root_path.exists():
        print(f"⚠️ 데이터 경로를 찾을 수 없음: {root_path}")
        return []

    sampled = []

    if cfg["nested"]:
        class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
        for class_dir in class_dirs:
            images = sorted(
                [p for p in class_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
            )
            if not images:
                continue
            chosen = images if len(images) <= samples_per_class else random.sample(images, samples_per_class)
            sampled.extend(chosen)
    else:
        # 단일 폴더(Kolektor)의 경우
        images = sorted(
            [p for p in root_path.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        )
        # 클래스 구분 없이 전체에서 샘플링 (필요시 수량 증대 가능)
        chosen = images if len(images) <= samples_per_class * 2 else random.sample(images, samples_per_class * 2)
        sampled.extend(chosen)

    return sorted(sampled)


def load_models():
    print("모델 로딩 중...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    vlm.to(DEVICE)
    vlm.eval()

    sbert = SentenceTransformer(SBERT_ID, device="cpu")
    return processor, vlm, sbert


def generate_response(image_path: Path, prompt: str, processor, vlm, gen_cfg: GenerationConfig) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = vlm.generate(
            **inputs,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            max_new_tokens=gen_cfg.max_new_tokens,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    return output_text.strip()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def compute_sbert_consistency(responses: List[str], sbert) -> Tuple[float, List[Dict]]:
    embeddings = sbert.encode(responses, convert_to_numpy=True, normalize_embeddings=True)
    pair_records = []
    scores = []

    for (i, emb_i), (j, emb_j) in itertools.combinations(enumerate(embeddings), 2):
        score = cosine_similarity(emb_i, emb_j)
        scores.append(score)
        pair_records.append({"pair": [i + 1, j + 1], "score": round(score, 6)})

    mean_score = float(np.mean(scores)) if scores else 0.0
    return mean_score, pair_records


def compute_bertscore_consistency(responses: List[str]) -> Tuple[Optional[float], List[Dict]]:
    if bertscore_score is None:
        return None, []

    pair_records = []
    scores = []

    for (i, a), (j, b) in itertools.combinations(enumerate(responses), 2):
        _, _, f1 = bertscore_score([a], [b], lang="en", verbose=False, rescale_with_baseline=True)
        score = float(f1[0].item())
        scores.append(score)
        pair_records.append({"pair": [i + 1, j + 1], "score": round(score, 6)})

    mean_score = float(np.mean(scores)) if scores else None
    return mean_score, pair_records


def records_to_dataframe(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {
            "experiment_name": rec["experiment_name"],
            "timestamp": rec["timestamp"],
            "prompt_group": rec["prompt_group"],
            "image_index": rec["image_index"],
            "image_path": rec["image_path"],
            "class_name": rec["class_name"],
            "sbert_mean_score": rec["sbert_mean_score"],
            "bertscore_mean_score": rec.get("bertscore_mean_score"),
        }
        for pr in rec["prompt_records"]:
            row[f"prompt_{pr['prompt_index']}"] = pr["prompt_text"]
            row[f"response_{pr['prompt_index']}"] = pr["response_text"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_prompt_group_experiment(
        image_paths: List[Path],
        prompt_group_name: str,
        prompts: List[str],
        processor, vlm, sbert, gen_cfg: GenerationConfig, output_jsonl_path: Path,
) -> List[Dict]:
    records = []
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for idx, image_path in enumerate(image_paths, start=1):
            class_name = image_path.parent.name
            responses = []
            prompt_records = []

            for prompt_idx, prompt in enumerate(prompts, start=1):
                response = generate_response(image_path, prompt, processor, vlm, gen_cfg)
                responses.append(response)
                prompt_records.append({
                    "prompt_index": prompt_idx, "prompt_text": prompt, "response_text": response
                })

            sbert_mean, sbert_pairs = compute_sbert_consistency(responses, sbert)

            if USE_BERTSCORE_FULL:
                bertscore_mean, bertscore_pairs = compute_bertscore_consistency(responses)
            else:
                bertscore_mean, bertscore_pairs = None, []

            record = {
                "experiment_name": EXPERIMENT_NAME,
                "timestamp": datetime.now().isoformat(),
                "prompt_group": prompt_group_name,
                "image_index": idx,
                "image_path": str(image_path),
                "class_name": class_name,
                "generation_config": asdict(gen_cfg),
                "prompt_records": prompt_records,
                "sbert_pairwise": sbert_pairs,
                "sbert_mean_score": round(sbert_mean, 6),
                "bertscore_pairwise": bertscore_pairs,
                "bertscore_mean_score": round(bertscore_mean, 6) if bertscore_mean is not None else None,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

            print(
                f"[{prompt_group_name}] {idx}/{len(image_paths)} | "
                f"{class_name} | {image_path.name} | SBERT={sbert_mean:.4f}"
                + (f" | BERTScore={bertscore_mean:.4f}" if bertscore_mean is not None else "")
            )
    return records


def summarize_records(records: List[Dict], metric_key: str) -> Dict:
    scores = [r[metric_key] for r in records if r.get(metric_key) is not None]
    summary = {
        "metric": metric_key,
        "count": len(scores),
        "mean": round(float(np.mean(scores)) if scores else 0.0, 6),
        "std": round(float(statistics.stdev(scores)) if len(scores) > 1 else 0.0, 6),
        "min": round(float(np.min(scores)) if scores else 0.0, 6),
        "max": round(float(np.max(scores)) if scores else 0.0, 6),
        "class_stats": [],
    }

    by_class = {}
    for r in records:
        if r.get(metric_key) is not None:
            by_class.setdefault(r["class_name"], []).append(r[metric_key])

    for class_name, vals in sorted(by_class.items()):
        summary["class_stats"].append({
            "class_name": class_name, "count": len(vals),
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(statistics.stdev(vals)) if len(vals) > 1 else 0.0, 6),
            "min": round(float(np.min(vals)), 6), "max": round(float(np.max(vals)), 6),
        })
    return summary


def plot_distributions(df: pd.DataFrame, metric_key: str, output_dir: Path) -> None:
    valid_df = df[df[metric_key].notna()].copy()
    if valid_df.empty: return

    plt.figure(figsize=(8, 5))
    for group in sorted(valid_df["prompt_group"].unique()):
        subset = valid_df[valid_df["prompt_group"] == group][metric_key]
        plt.hist(subset, bins=15, alpha=0.5, label=group)
    plt.xlabel(metric_key)
    plt.ylabel("count")
    plt.title(f"{metric_key} histogram by prompt group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric_key}_histogram.png", dpi=200)
    plt.close()


def save_candidate_manifests(records: List[Dict], metric_key: str, output_dir: Path, prefix: str) -> Dict:
    valid_records = [r for r in records if r.get(metric_key) is not None]
    sorted_records = sorted(valid_records, key=lambda x: x[metric_key])

    low_records = sorted_records[:LOW_K]
    high_records = sorted_records[-HIGH_K:]
    random_records = random.sample(valid_records, min(RANDOM_K, len(valid_records)))

    groups = {"low": low_records, "high": high_records, "random": random_records}
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for group_name, recs in groups.items():
        txt_path = output_dir / f"{prefix}_{metric_key}_{group_name}_images.txt"
        json_path = output_dir / f"{prefix}_{metric_key}_{group_name}_records.json"
        with open(txt_path, "w", encoding="utf-8") as f:
            for rec in recs: f.write(f"{rec['image_path']}\n")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        result[group_name] = recs
    return result


def posthoc_bertscore_for_candidates(candidate_records: List[Dict], output_path: Path) -> List[Dict]:
    if not USE_BERTSCORE_CANDIDATES or bertscore_score is None:
        return candidate_records

    updated = []
    for idx, rec in enumerate(candidate_records, start=1):
        responses = [p["response_text"] for p in rec["prompt_records"]]
        mean_score, pairwise = compute_bertscore_consistency(responses)
        new_rec = dict(rec)
        new_rec["posthoc_bertscore_mean_score"] = round(mean_score, 6) if mean_score is not None else None
        new_rec["posthoc_bertscore_pairwise"] = pairwise
        updated.append(new_rec)
        print(
            f"[posthoc BERTScore] {idx}/{len(candidate_records)} | {rec['prompt_group']} | {rec['class_name']} | {new_rec['posthoc_bertscore_mean_score']}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
    return updated


def save_source_manifest(output_path: Path, gen_cfg: GenerationConfig, sampled_images: List[Path],
                         dataset_name: str) -> None:
    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset_name": dataset_name,
        "created_at": datetime.now().isoformat(),
        "generation_config": asdict(gen_cfg),
        "sampled_images_count": len(sampled_images),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def save_json(data: Dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


# =============================================================================
# [Main Execution Loop]
# =============================================================================
def main():
    set_seed(SEED)
    processor, vlm, sbert = load_models()

    gen_cfg = GenerationConfig(
        device=DEVICE, model_id=MODEL_ID, sbert_id=SBERT_ID, seed=SEED,
        do_sample=DO_SAMPLE, temperature=TEMPERATURE, top_p=TOP_P, max_new_tokens=MAX_NEW_TOKENS,
    )

    # 정의된 데이터셋들을 순회하며 기존 실험 로직을 완벽하게 반복 수행
    for ds_name, ds_cfg in DATASET_CONFIGS.items():
        print(f"\n{'=' * 80}")
        print(f"🚀 [{ds_name}] 데이터셋 실험 시작")
        print(f"{'=' * 80}")

        dirs = prepare_output_dirs(dataset_name=ds_name)
        image_paths = sample_images_universal(ds_name, SAMPLES_PER_CLASS)

        if not image_paths:
            print(f"[{ds_name}] 건너뜀: 조건에 맞는 이미지 파일이 없습니다.")
            continue

        save_source_manifest(dirs["manifest"] / "source_manifest.json", gen_cfg, image_paths, ds_name)

        rephrase_records = run_prompt_group_experiment(
            image_paths=image_paths,
            prompt_group_name="rephrase",
            prompts=REPHRASE_PROMPTS,
            processor=processor, vlm=vlm, sbert=sbert,
            gen_cfg=gen_cfg,
            output_jsonl_path=dirs["jsonl"] / "rephrase_results.jsonl"
        )

        viewpoint_records = run_prompt_group_experiment(
            image_paths=image_paths,
            prompt_group_name="viewpoint",
            prompts=VIEWPOINT_PROMPTS,
            processor=processor, vlm=vlm, sbert=sbert,
            gen_cfg=gen_cfg,
            output_jsonl_path=dirs["jsonl"] / "viewpoint_results.jsonl"
        )

        all_records = rephrase_records + viewpoint_records
        df = records_to_dataframe(all_records)
        save_csv(df, dirs["summary"] / "all_results.csv")

        sbert_summary = {
            "rephrase": summarize_records(rephrase_records, "sbert_mean_score"),
            "viewpoint": summarize_records(viewpoint_records, "sbert_mean_score"),
        }
        save_json(sbert_summary, dirs["summary"] / "sbert_mean_score_summary.json")
        plot_distributions(df, "sbert_mean_score", dirs["plots"])

        rephrase_candidates = save_candidate_manifests(rephrase_records, "sbert_mean_score", dirs["candidates"],
                                                       prefix="rephrase")
        viewpoint_candidates = save_candidate_manifests(viewpoint_records, "sbert_mean_score", dirs["candidates"],
                                                        prefix="viewpoint")

        candidate_pool = (
                rephrase_candidates.get("low", []) + rephrase_candidates.get("high", []) +
                viewpoint_candidates.get("low", []) + viewpoint_candidates.get("high", [])
        )

        if candidate_pool:
            posthoc_bertscore_for_candidates(candidate_pool, dirs["summary"] / "posthoc_bertscore_candidates.json")

        final_compare = {
            "experiment_name": EXPERIMENT_NAME,
            "dataset_name": ds_name,
            "created_at": datetime.now().isoformat(),
            "generation_config": asdict(gen_cfg),
            "sbert_summary": sbert_summary,
        }
        save_json(final_compare, dirs["summary"] / "final_compare.json")

        print(f"\n✅ [{ds_name}] 완료! 결과물이 저장된 폴더: {dirs['base']}")

    print("\n" + "=" * 80)
    print("🎉 모든 데이터셋 통합 실험이 성공적으로 종료되었습니다.")
    print("=" * 80)


if __name__ == "__main__":
    main()