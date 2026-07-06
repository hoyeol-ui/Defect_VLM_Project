
import csv
import itertools
import json
import random
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bert_score import score as bertscore_score
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# =============================================================================
# [Upstream / Original Sources]
# =============================================================================
# Qwen2-VL model card:
# https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
#
# Qwen2-VL official Transformers docs:
# https://huggingface.co/docs/transformers/model_doc/qwen2_vl
#
# Qwen2-VL modeling source (Transformers upstream):
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
#
# qwen_vl_utils / process_vision_info upstream reference:
# https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
#
# Sentence-Transformers model card (all-MiniLM-L6-v2):
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#
# SentenceTransformer API docs:
# https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
#
# BERTScore official repository:
# https://github.com/Tiiiger/bert_score
#
# BERTScore paper:
# https://openreview.net/forum?id=SkeHuCVFDr
#
# Transformers processors docs:
# https://huggingface.co/docs/transformers/en/main_classes/processors
# https://huggingface.co/docs/transformers/en/model_doc/auto
#
# PyTorch MPS backend docs:
# https://docs.pytorch.org/docs/stable/notes/mps.html
# https://developer.apple.com/metal/pytorch/
#
# Matplotlib histogram docs:
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
#
# Seaborn boxplot docs:
# https://seaborn.pydata.org/generated/seaborn.boxplot.html
#
# pandas DataFrame.to_csv docs:
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
# =============================================================================


# =============================================================================
# [1] 실험 설정
# =============================================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SBERT_ID = "sentence-transformers/all-MiniLM-L6-v2"

EXPERIMENT_NAME = "consistency_v2_rephrase_viewpoint_sbert_bertscore"

# 기존 파일럿보다 확대
SAMPLES_PER_CLASS = 10
SEED = 42

DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 64

LOW_K = 10
HIGH_K = 10
RANDOM_K = 10

DATASET_RELATIVE_PATH = Path("data") / "NEU-DET" / "train" / "images"

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


# -----------------------------------------------------------------------------
# Reference:
# - Python random docs: https://docs.python.org/3/library/random.html
# - PyTorch MPS docs: https://docs.pytorch.org/docs/stable/notes/mps.html
#
# Rationale:
# - 교수님 피드백상 seed 통제는 필수에 가깝다.
# - 본 함수는 재현성을 위해 Python / NumPy / Torch seed를 고정한다.
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# Reference:
# - pathlib official docs: https://docs.python.org/3/library/pathlib.html
#
# Rationale:
# - 프로젝트 루트를 기준으로 data, logs 경로를 안정적으로 계산하기 위한 유틸.
# -----------------------------------------------------------------------------
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Reference:
# - pathlib official docs: https://docs.python.org/3/library/pathlib.html
# - json official docs: https://docs.python.org/3/library/json.html
#
# Rationale:
# - 실험 재현성을 위해 결과/요약/그림/manifest 경로를 구조화해서 저장.
# -----------------------------------------------------------------------------
def prepare_output_dirs() -> Dict[str, Path]:
    root = get_project_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = root / "logs" / EXPERIMENT_NAME / timestamp

    dirs = {
        "base": base_dir,
        "jsonl": base_dir / "jsonl",
        "summary": base_dir / "summary",
        "plots": base_dir / "plots",
        "manifest": base_dir / "manifest",
    }

    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    return dirs


# -----------------------------------------------------------------------------
# Reference:
# - Python random.sample docs: https://docs.python.org/3/library/random.html
# - pathlib official docs: https://docs.python.org/3/library/pathlib.html
#
# Rationale:
# - 클래스별 균등 샘플링으로 분포를 보기 위한 파일럿-확장 실험 구성.
# - 아직 최종 논문용 split 고정이 아니라 "본 실험 전 단계"라는 점을 코드에 남기는 것이 안전.
# -----------------------------------------------------------------------------
def sample_images_by_class(samples_per_class: int) -> List[Path]:
    image_root = get_project_root() / DATASET_RELATIVE_PATH
    if not image_root.exists():
        raise FileNotFoundError(f"데이터 경로를 찾을 수 없음: {image_root}")

    class_dirs = sorted([d for d in image_root.iterdir() if d.is_dir()])
    sampled = []

    for class_dir in class_dirs:
        images = sorted(
            [p for p in class_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        )
        if not images:
            continue

        if len(images) <= samples_per_class:
            chosen = images
        else:
            chosen = random.sample(images, samples_per_class)

        sampled.extend(chosen)

    return sorted(sampled)


# -----------------------------------------------------------------------------
# Reference:
# - Qwen2-VL model card: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
# - Qwen2-VL docs: https://huggingface.co/docs/transformers/model_doc/qwen2_vl
# - Transformers auto classes: https://huggingface.co/docs/transformers/en/model_doc/auto
# - SentenceTransformer API: https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
#
# Rationale:
# - Qwen2-VL + AutoProcessor + SBERT를 로드.
# - 여기서는 안정성 우선이라 공격적인 mixed precision 최적화는 하지 않음.
# -----------------------------------------------------------------------------
def load_models():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID)
    vlm.to(DEVICE)
    vlm.eval()

    sbert = SentenceTransformer(SBERT_ID, device=DEVICE)
    return processor, vlm, sbert


# -----------------------------------------------------------------------------
# Reference:
# - Qwen2-VL docs example flow:
#   https://huggingface.co/docs/transformers/model_doc/qwen2_vl
# - Qwen2-VL modeling source:
#   https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# - process_vision_info upstream:
#   https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
#
# Borrowed experimental logic:
# - apply_chat_template -> process_vision_info -> processor(...) -> model.generate(...)
# - 이 흐름은 공식 문서/원본 구현 사용 흐름을 따라 차용했으며,
#   본 연구 목적에 맞게 prompt-group 실험 구조로 재조합함.
# -----------------------------------------------------------------------------
def generate_response(
    image_path: Path,
    prompt: str,
    processor,
    vlm,
    gen_cfg: GenerationConfig,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return output_text.strip()


# -----------------------------------------------------------------------------
# Reference:
# - NumPy norm docs: https://numpy.org/devdocs/reference/generated/numpy.linalg.norm.html
#
# Rationale:
# - 기존 np.dot(a, b) 단독 사용보다, 정규화된 cosine similarity를 명시적으로 계산.
# -----------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


# -----------------------------------------------------------------------------
# Reference:
# - SentenceTransformer API docs:
#   https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
# - all-MiniLM-L6-v2 model card:
#   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#
# Borrowed experimental logic:
# - 사용자 기존 실험의 핵심 지표(SBERT cosine)를 유지하되,
#   정규화 cosine 및 pairwise 평균으로 명시화.
# -----------------------------------------------------------------------------
def compute_sbert_consistency(responses: List[str], sbert) -> Tuple[float, List[Dict]]:
    embeddings = sbert.encode(responses, convert_to_numpy=True)

    pair_records = []
    scores = []

    for (i, emb_i), (j, emb_j) in itertools.combinations(enumerate(embeddings), 2):
        score = cosine_similarity(emb_i, emb_j)
        scores.append(score)
        pair_records.append(
            {
                "pair": [i + 1, j + 1],
                "score": round(score, 6),
            }
        )

    mean_score = float(np.mean(scores)) if scores else 0.0
    return mean_score, pair_records


# -----------------------------------------------------------------------------
# Reference:
# - BERTScore official repo: https://github.com/Tiiiger/bert_score
# - BERTScore paper: https://openreview.net/forum?id=SkeHuCVFDr
#
# Rationale:
# - 교수님 피드백 3번(SBERT 외 비교 지표)을 최소 수준으로 반영.
# - 현재 구현은 pairwise F1 평균을 사용.
# - 아직 NLI는 넣지 않았고, SBERT + BERTScore까지만 반영.
# -----------------------------------------------------------------------------
def compute_bertscore_consistency(responses: List[str]) -> Tuple[float, List[Dict]]:
    pair_records = []
    scores = []

    for (i, a), (j, b) in itertools.combinations(enumerate(responses), 2):
        P, R, F1 = bertscore_score(
            [a],
            [b],
            lang="en",
            verbose=False,
            rescale_with_baseline=True,
        )
        score = float(F1[0].item())
        scores.append(score)
        pair_records.append(
            {
                "pair": [i + 1, j + 1],
                "score": round(score, 6),
            }
        )

    mean_score = float(np.mean(scores)) if scores else 0.0
    return mean_score, pair_records


# -----------------------------------------------------------------------------
# Reference:
# - pandas to_csv docs:
#   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
#
# Rationale:
# - 논문 표/부록/후속 분석을 위해 row 기반 flat table 생성.
# -----------------------------------------------------------------------------
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
            "bertscore_mean_score": rec["bertscore_mean_score"],
        }

        for pr in rec["prompt_records"]:
            row[f"prompt_{pr['prompt_index']}"] = pr["prompt_text"]
            row[f"response_{pr['prompt_index']}"] = pr["response_text"]

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Reference:
# - Python json docs: https://docs.python.org/3/library/json.html
#
# Rationale:
# - 실험 단위 결과(JSONL) 저장.
# - prompt group, image path, responses, metric details를 모두 남겨서
#   "어떤 설정에서 어떤 값이 나왔는지"를 추적 가능하게 함.
# -----------------------------------------------------------------------------
def run_prompt_group_experiment(
    image_paths: List[Path],
    prompt_group_name: str,
    prompts: List[str],
    processor,
    vlm,
    sbert,
    gen_cfg: GenerationConfig,
    output_jsonl_path: Path,
) -> List[Dict]:
    all_records = []

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for idx, image_path in enumerate(image_paths, start=1):
            class_name = image_path.parent.name

            responses = []
            prompt_records = []

            for prompt_idx, prompt in enumerate(prompts, start=1):
                response = generate_response(
                    image_path=image_path,
                    prompt=prompt,
                    processor=processor,
                    vlm=vlm,
                    gen_cfg=gen_cfg,
                )
                responses.append(response)
                prompt_records.append(
                    {
                        "prompt_index": prompt_idx,
                        "prompt_text": prompt,
                        "response_text": response,
                    }
                )

            sbert_mean, sbert_pairs = compute_sbert_consistency(responses, sbert)
            bertscore_mean, bertscore_pairs = compute_bertscore_consistency(responses)

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
                "bertscore_mean_score": round(bertscore_mean, 6),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_records.append(record)

            print(
                f"[{prompt_group_name}] {idx}/{len(image_paths)} | "
                f"{class_name} | {image_path.name} | "
                f"SBERT={sbert_mean:.4f} | BERTScore={bertscore_mean:.4f}"
            )

    return all_records


# -----------------------------------------------------------------------------
# Reference:
# - Python statistics module docs:
#   https://docs.python.org/3/library/statistics.html
#
# Rationale:
# - 전체 평균/표준편차/범위, 클래스별 통계 요약.
# - 교수님 피드백 1번의 "전체 분포, 클래스별 평균·분산 확인" 대응.
# -----------------------------------------------------------------------------
def summarize_records(records: List[Dict], metric_key: str) -> Dict:
    scores = [r[metric_key] for r in records]

    summary = {
        "metric": metric_key,
        "count": len(scores),
        "mean": round(float(np.mean(scores)) if scores else 0.0, 6),
        "std": round(float(statistics.stdev(scores)) if len(scores) > 1 else 0.0, 6),
        "min": round(float(np.min(scores)) if scores else 0.0, 6),
        "max": round(float(np.max(scores)) if scores else 0.0, 6),
        "range": round((float(np.max(scores)) - float(np.min(scores))) if scores else 0.0, 6),
        "class_stats": [],
    }

    by_class = {}
    for r in records:
        by_class.setdefault(r["class_name"], []).append(r[metric_key])

    for class_name, vals in sorted(by_class.items()):
        summary["class_stats"].append(
            {
                "class_name": class_name,
                "count": len(vals),
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(statistics.stdev(vals)) if len(vals) > 1 else 0.0, 6),
                "min": round(float(np.min(vals)), 6),
                "max": round(float(np.max(vals)), 6),
            }
        )

    return summary


# -----------------------------------------------------------------------------
# Reference:
# - Matplotlib histogram docs:
#   https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
# - Seaborn boxplot docs:
#   https://seaborn.pydata.org/generated/seaborn.boxplot.html
#
# Rationale:
# - 교수님께 "분포를 확인했다"를 말하려면 숫자만으로는 부족함.
# - histogram / classwise boxplot / prompt-group summary plot 생성.
# -----------------------------------------------------------------------------
def plot_distributions(df: pd.DataFrame, metric_key: str, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    # 1) 전체 histogram
    plt.figure(figsize=(8, 5))
    for group in sorted(df["prompt_group"].unique()):
        subset = df[df["prompt_group"] == group][metric_key]
        plt.hist(subset, bins=15, alpha=0.5, label=group)
    plt.xlabel(metric_key)
    plt.ylabel("count")
    plt.title(f"{metric_key} histogram by prompt group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric_key}_histogram.png", dpi=200)
    plt.close()

    # 2) 클래스별 boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="class_name", y=metric_key, hue="prompt_group")
    plt.xticks(rotation=30)
    plt.title(f"{metric_key} by class and prompt group")
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric_key}_classwise_boxplot.png", dpi=200)
    plt.close()

    # 3) prompt-group 평균 bar plot
    plt.figure(figsize=(6, 5))
    group_mean = df.groupby("prompt_group")[metric_key].mean().reset_index()
    sns.barplot(data=group_mean, x="prompt_group", y=metric_key)
    plt.title(f"{metric_key} mean by prompt group")
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric_key}_group_mean_bar.png", dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Reference:
# - Python random.sample docs: https://docs.python.org/3/library/random.html
# - pathlib docs: https://docs.python.org/3/library/pathlib.html
#
# Rationale:
# - AL 실험 전 단계로 low/high/random 후보 목록을 텍스트 manifest로 저장.
# - 지금은 아직 학습 성능 검증이 아니라, candidate set 구성 단계.
# -----------------------------------------------------------------------------
def save_candidate_manifests(records: List[Dict], metric_key: str, output_dir: Path) -> None:
    sorted_records = sorted(records, key=lambda x: x[metric_key])

    low_records = sorted_records[:LOW_K]
    high_records = sorted_records[-HIGH_K:]
    random_records = random.sample(records, min(RANDOM_K, len(records)))

    groups = {
        "low": low_records,
        "high": high_records,
        "random": random_records,
    }

    for group_name, recs in groups.items():
        txt_path = output_dir / f"{metric_key}_{group_name}_consistency_images.txt"
        json_path = output_dir / f"{metric_key}_{group_name}_consistency_images.json"

        with open(txt_path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(f"{rec['image_path']}\n")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Reference:
# - Python json docs: https://docs.python.org/3/library/json.html
#
# Rationale:
# - 어떤 upstream를 참고했고, 어떤 실험 조건으로, 어떤 이미지 샘플을 썼는지
#   코드 단계에서 남기는 provenance 파일.
# -----------------------------------------------------------------------------
def save_source_manifest(output_path: Path, gen_cfg: GenerationConfig, sampled_images: List[Path]) -> None:
    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "created_at": datetime.now().isoformat(),
        "code_statement": (
            "This script is an experiment-specific adaptation that references official upstream "
            "model cards, API docs, and source code, rather than a verbatim copy of any single source file."
        ),
        "generation_config": asdict(gen_cfg),
        "dataset_relative_path": str(DATASET_RELATIVE_PATH),
        "sampled_images_count": len(sampled_images),
        "sampled_images": [str(p) for p in sampled_images],
        "upstream_sources": {
            "qwen_model_card": "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct",
            "qwen_docs": "https://huggingface.co/docs/transformers/model_doc/qwen2_vl",
            "qwen_modeling_source": "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py",
            "qwen_vl_utils_source": "https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py",
            "sbert_model_card": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
            "sbert_api_docs": "https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html",
            "bertscore_repo": "https://github.com/Tiiiger/bert_score",
            "bertscore_paper": "https://openreview.net/forum?id=SkeHuCVFDr",
            "mps_docs": "https://docs.pytorch.org/docs/stable/notes/mps.html",
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Reference:
# - pandas to_csv docs:
#   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
#
# Rationale:
# - 논문 표, 부록, 후속 분석에 바로 쓸 수 있는 CSV 저장.
# -----------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# Reference:
# - Python json docs: https://docs.python.org/3/library/json.html
#
# Rationale:
# - metric별 summary를 JSON으로 저장해 실험 요약 기록을 구조화.
# -----------------------------------------------------------------------------
def save_json(data: Dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Reference:
# - 전체 함수는 위에 명시된 upstream docs / sources를 조합해
#   본 연구 목적에 맞게 orchestration한 엔트리 포인트.
#
# Rationale:
# - 오늘 단계의 목적은 "최종 AL 검증"이 아니라
#   "consistency 분포와 prompt 민감도"를 본 실험 수준으로 정리하는 것.
# -----------------------------------------------------------------------------
def main():
    set_seed(SEED)
    dirs = prepare_output_dirs()

    gen_cfg = GenerationConfig(
        device=DEVICE,
        model_id=MODEL_ID,
        sbert_id=SBERT_ID,
        seed=SEED,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    image_paths = sample_images_by_class(SAMPLES_PER_CLASS)
    save_source_manifest(dirs["manifest"] / "source_manifest.json", gen_cfg, image_paths)

    processor, vlm, sbert = load_models()

    # rephrase 실험
    rephrase_records = run_prompt_group_experiment(
        image_paths=image_paths,
        prompt_group_name="rephrase",
        prompts=REPHRASE_PROMPTS,
        processor=processor,
        vlm=vlm,
        sbert=sbert,
        gen_cfg=gen_cfg,
        output_jsonl_path=dirs["jsonl"] / "rephrase_results.jsonl",
    )

    # viewpoint 실험
    viewpoint_records = run_prompt_group_experiment(
        image_paths=image_paths,
        prompt_group_name="viewpoint",
        prompts=VIEWPOINT_PROMPTS,
        processor=processor,
        vlm=vlm,
        sbert=sbert,
        gen_cfg=gen_cfg,
        output_jsonl_path=dirs["jsonl"] / "viewpoint_results.jsonl",
    )

    all_records = rephrase_records + viewpoint_records
    df = records_to_dataframe(all_records)

    save_csv(df, dirs["summary"] / "all_results.csv")

    # metric별 요약 저장
    for metric_key in ["sbert_mean_score", "bertscore_mean_score"]:
        summary = {
            "rephrase": summarize_records(rephrase_records, metric_key),
            "viewpoint": summarize_records(viewpoint_records, metric_key),
        }
        save_json(summary, dirs["summary"] / f"{metric_key}_summary.json")
        save_candidate_manifests(all_records, metric_key, dirs["manifest"])
        plot_distributions(df, metric_key, dirs["plots"])

    print("=" * 80)
    print("실험 완료")
    print(f"output_dir: {dirs['base']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
