# =============================================================================
# Upstream / Original Sources
# =============================================================================
# [Qwen2-VL model card]
# https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
#
# [Qwen2-VL official Transformers documentation]
# https://huggingface.co/docs/transformers/model_doc/qwen2_vl
#
# [Qwen2-VL modeling source in Hugging Face Transformers]
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
#
# [process_vision_info upstream reference]
# https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
#
# [Sentence-Transformers model card: all-MiniLM-L6-v2]
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#
# [SentenceTransformer API docs]
# https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
#
# [Transformers processors / auto classes docs]
# https://huggingface.co/docs/transformers/en/main_classes/processors
# https://huggingface.co/docs/transformers/en/model_doc/auto
#
# [PyTorch MPS backend docs]
# https://docs.pytorch.org/docs/stable/notes/mps.html
# https://developer.apple.com/metal/pytorch/
# =============================================================================


import csv
import itertools
import json
import math
import random
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# 사용자 원본 코드 조각에 존재했던 유틸
# 출처: consistency_prompt_ensemble.py / consistency_mixed_eval.py
from qwen_vl_utils import process_vision_info


# =============================================================================
# [1] 실행 환경 / 실험 설정
# =============================================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SBERT_ID = "all-MiniLM-L6-v2"

# 오늘은 파일럿보다 표본 수를 늘리는 것이 목적
# 기존 코드 조각에서는 SAMPLES_PER_CLASS = 2 였음
# 출처: consistency_prompt_ensemble.py, consistency_mixed_eval.py
SAMPLES_PER_CLASS = 10

# 실험 재현성을 위해 seed를 고정
SEED = 42

# 생성 조건을 명시적으로 남긴다
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 64

# 출력 디렉토리 이름
EXPERIMENT_NAME = "today_consistency_rephrase_vs_viewpoint"

# 데이터 경로
# 로그에서 관찰된 프로젝트 구조를 따르되, 실제 환경에 맞게 수정 가능
DATASET_RELATIVE_PATH = Path("data") / "NEU-DET" / "train" / "images"

# 오늘 실험의 핵심:
# 1) 같은 의미의 재표현(rephrase)
# 2) 관점을 바꾼 질문(viewpoint)
REPHRASE_PROMPTS = [
    "Describe the visible defect on this metal surface in one short sentence.",
    "In one short sentence, describe the defect visible on this metal surface.",
    "Briefly describe the visible defect on the metal surface in one sentence.",
]

VIEWPOINT_PROMPTS = [
    "Describe the defect by focusing on its location and affected area in one short sentence.",
    "Describe the defect by focusing on its shape or structural pattern in one short sentence.",
    "Describe the defect by focusing on its texture, surface condition, or appearance in one short sentence.",
]


# =============================================================================
# [2] 메타데이터 / provenance 구조
# =============================================================================

@dataclass
class SourceRecord:
    """출처 기록용 구조체"""
    label: str
    kind: str
    url: str
    note: str


@dataclass
class GenerationConfig:
    """생성 조건 기록용 구조체"""
    device: str
    model_id: str
    sbert_id: str
    seed: int
    do_sample: bool
    temperature: float
    top_p: float
    max_new_tokens: int


def build_source_manifest() -> List[SourceRecord]:
    """
    오늘 실험 코드의 출처/근거 문서를 한곳에 모은다.
    논문/보고서 작성 시 provenance(출처 추적) 목적으로 저장한다.
    """
    return [
        SourceRecord(
            label="consistency_prompt_ensemble.py",
            kind="user_source_code",
            url="https://www.genspark.ai/api/files/s/Af2UtGLb",
            note="Qwen2-VL + SBERT + prompt ensemble 기반 기존 실험 코드 조각",
        ),
        SourceRecord(
            label="consistency_mixed_eval.py",
            kind="user_source_code",
            url="https://www.genspark.ai/api/files/s/N1X9N2hm",
            note="repeat generation baseline / 클래스 균등 샘플링 구조 참고",
        ),
        SourceRecord(
            label="consistency_full_analysis.py",
            kind="user_source_code",
            url="https://www.genspark.ai/api/files/s/Z1SXt4Yt",
            note="전체 통계와 low/mid/high consistency 요약 구조 참고",
        ),
        SourceRecord(
            label="select_low_consistency_samples.py",
            kind="user_source_code",
            url="https://www.genspark.ai/api/files/s/wwxvnRWy",
            note="low consistency 샘플 추출 후처리 흐름 참고",
        ),
        SourceRecord(
            label="consistency_run_20260329_130515.txt",
            kind="user_result_log",
            url="https://www.genspark.ai/api/files/s/49twxFG6",
            note="동일/유사 반복 프롬프트 baseline에서 1.0000 수렴 경향 확인",
        ),
        SourceRecord(
            label="mixed_eval_20260329_131015.txt",
            kind="user_result_log",
            url="https://www.genspark.ai/api/files/s/TExFFH77",
            note="repeat generation baseline의 한계 확인",
        ),
        SourceRecord(
            label="FULL_analysis_20260329_140129.txt",
            kind="user_result_log",
            url="https://www.genspark.ai/api/files/s/nyhNN6EU",
            note="prompt ensemble 이후 score 분포 형성 확인, 단 표본 수는 작음",
        ),
        SourceRecord(
            label="설명 일관성 기반 능동학습 제조 결함 탐지 VLM 연구계획서 심층 타당성 검토",
            kind="research_feedback_doc",
            url="https://www.genspark.ai/api/files/s/M7HcjKs2",
            note="교수님 피드백 반영 핵심: 조건 분리, 통제 실험, seed 및 sampling 설정 명시",
        ),
        SourceRecord(
            label="학위논문 연구계획서_2025002034 이호열 버전2",
            kind="research_plan_doc",
            url="https://www.genspark.ai/api/files/s/S6TvQD5Y",
            note="consistency -> uncertainty proxy -> AL 검증 구조의 상위 계획",
        ),
        SourceRecord(
            label="260330 랩미팅 발표자료 대본",
            kind="meeting_script",
            url="https://www.genspark.ai/api/files/s/2ZvQUeHU",
            note="파일럿 실험 구조, SBERT cosine 사용 근거, 현재 한계 정리",
        ),
    ]


# =============================================================================
# [3] 유틸
# =============================================================================

def set_seed(seed: int) -> None:
    """
    재현성 확보용 seed 고정
    교수님 피드백상 seed 통제는 필수에 가깝다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        # MPS에서도 torch manual seed는 걸어두는 편이 낫다.
        torch.manual_seed(seed)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    SBERT 임베딩 코사인 유사도 계산.
    기존 사용자 코드 조각에서는 np.dot(a, b) 형태가 보였으나,
    여기서는 안전하게 정규화 후 코사인을 계산한다.
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def get_project_root() -> Path:
    """
    원본 코드 조각 스타일 유지
    출처: consistency_prompt_ensemble.py / consistency_mixed_eval.py / consistency_full_analysis.py
    """
    return Path(__file__).resolve().parent.parent


def prepare_output_dirs() -> Dict[str, Path]:
    """
    로그/리포트/manifest 저장 경로 생성
    """
    root = get_project_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = root / "logs" / EXPERIMENT_NAME / timestamp
    jsonl_dir = base_dir / "jsonl"
    summary_dir = base_dir / "summary"
    manifest_dir = base_dir / "manifest"

    jsonl_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_dir,
        "jsonl": jsonl_dir,
        "summary": summary_dir,
        "manifest": manifest_dir,
    }


def resolve_image_root() -> Path:
    """
    데이터셋 루트 경로 해결
    """
    root = get_project_root()
    image_root = root / DATASET_RELATIVE_PATH
    if not image_root.exists():
        raise FileNotFoundError(
            f"[오류] 데이터 경로를 찾지 못했습니다: {image_root}\n"
            f"- DATASET_RELATIVE_PATH 또는 프로젝트 루트를 확인하세요."
        )
    return image_root


def sample_images_by_class(samples_per_class: int) -> List[Path]:
    """
    클래스별 균등 샘플링
    기존 사용자 코드 스타일 유지

    주의:
    - 지금은 논문 최종본이 아니라 오늘 실험용이므로 random seed 기반 샘플링
    - 추후에는 train/val/test split 고정, image manifest 고정 권장
    """
    image_root = resolve_image_root()
    class_dirs = sorted([d for d in image_root.iterdir() if d.is_dir()])

    sampled_paths = []
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

        sampled_paths.extend(chosen)

    return sorted(sampled_paths)


def infer_class_name(image_path: Path) -> str:
    """
    로그 구조 기준으로 부모 디렉토리를 클래스명으로 사용
    예: .../images/inclusion/inclusion_1.jpg -> inclusion
    """
    return image_path.parent.name


# =============================================================================
# [4] 모델 로드
# =============================================================================

def load_models():
    """
    Qwen2-VL + Processor + SBERT 로드

    주의:
    - MPS 환경에서 fp16/bfloat16 안정성 이슈가 있을 수 있어 기본 dtype은 명시적으로 강제하지 않음
    - 교수님 피드백상 MPS 환경은 "안정성 우선"이므로 today experiment에서는 보수적으로 간다.
    """
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID)
    vlm.to(DEVICE)
    vlm.eval()

    sbert = SentenceTransformer(SBERT_ID, device=DEVICE)

    return processor, vlm, sbert


# =============================================================================
# [5] Qwen 응답 생성
# =============================================================================

def generate_response(
    image_path: Path,
    prompt: str,
    processor,
    vlm,
    gen_cfg: GenerationConfig,
) -> str:
    """
    이미지 + 프롬프트로 Qwen2-VL 응답 생성
    """

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

    # 입력 부분 제거 후 디코딩
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return output_text.strip()


# =============================================================================
# [6] consistency 계산
# =============================================================================

def compute_pairwise_consistency(responses: List[str], sbert) -> Tuple[float, List[Dict]]:
    """
    응답 리스트에 대해 pairwise cosine similarity 평균을 계산한다.
    """
    embeddings = sbert.encode(responses, convert_to_numpy=True)

    pair_records = []
    scores = []

    for (i, emb_i), (j, emb_j) in itertools.combinations(enumerate(embeddings), 2):
        score = cosine_similarity(emb_i, emb_j)
        scores.append(score)
        pair_records.append(
            {
                "pair": [i + 1, j + 1],
                "cosine_similarity": round(score, 6),
            }
        )

    mean_score = float(np.mean(scores)) if scores else 0.0
    return mean_score, pair_records


# =============================================================================
# [7] 실험 단위 실행
# =============================================================================

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
    """
    하나의 prompt group(rephrase 또는 viewpoint)에 대해 전체 이미지 실험 수행
    """
    all_records = []

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for idx, image_path in enumerate(image_paths, start=1):
            class_name = infer_class_name(image_path)

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

            mean_consistency_score, pairwise = compute_pairwise_consistency(responses, sbert)

            record = {
                "experiment_name": EXPERIMENT_NAME,
                "timestamp": datetime.now().isoformat(),
                "prompt_group": prompt_group_name,
                "image_index": idx,
                "image_path": str(image_path),
                "class_name": class_name,
                "num_prompts": len(prompts),
                "generation_config": asdict(gen_cfg),
                "prompt_records": prompt_records,
                "pairwise_similarity": pairwise,
                "mean_consistency_score": round(mean_consistency_score, 6),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_records.append(record)

            print(
                f"[{prompt_group_name}] {idx}/{len(image_paths)} | "
                f"{class_name} | {image_path.name} | score={mean_consistency_score:.4f}"
            )

    return all_records


# =============================================================================
# [8] 분석 / 요약
# =============================================================================

def safe_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def summarize_records(records: List[Dict]) -> Dict:
    """
    전체 및 클래스별 통계 산출
    """
    scores = [r["mean_consistency_score"] for r in records]
    summary = {
        "count": len(scores),
        "mean": round(float(np.mean(scores)) if scores else 0.0, 6),
        "std": round(safe_std(scores), 6),
        "min": round(float(np.min(scores)) if scores else 0.0, 6),
        "max": round(float(np.max(scores)) if scores else 0.0, 6),
        "range": round((float(np.max(scores)) - float(np.min(scores))) if scores else 0.0, 6),
        "class_stats": [],
    }

    by_class: Dict[str, List[float]] = {}
    for r in records:
        by_class.setdefault(r["class_name"], []).append(r["mean_consistency_score"])

    for class_name, class_scores in sorted(by_class.items()):
        summary["class_stats"].append(
            {
                "class_name": class_name,
                "count": len(class_scores),
                "mean": round(float(np.mean(class_scores)), 6),
                "std": round(safe_std(class_scores), 6),
                "min": round(float(np.min(class_scores)), 6),
                "max": round(float(np.max(class_scores)), 6),
            }
        )

    return summary


def top_k_records(records: List[Dict], k: int, reverse: bool = False) -> List[Dict]:
    """
    reverse=False -> low consistency 상위 k개
    reverse=True  -> high consistency 상위 k개
    """
    return sorted(records, key=lambda x: x["mean_consistency_score"], reverse=reverse)[:k]


def write_summary_txt(
    prompt_group_name: str,
    summary: Dict,
    low_records: List[Dict],
    high_records: List[Dict],
    output_path: Path,
) -> None:
    """
    사람이 바로 읽을 수 있는 TXT 요약 저장
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"EXPERIMENT: {EXPERIMENT_NAME}\n")
        f.write(f"PROMPT_GROUP: {prompt_group_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"count: {summary['count']}\n")
        f.write(f"mean: {summary['mean']}\n")
        f.write(f"std: {summary['std']}\n")
        f.write(f"min: {summary['min']}\n")
        f.write(f"max: {summary['max']}\n")
        f.write(f"range: {summary['range']}\n")
        f.write("\n")

        f.write("[CLASS STATS]\n")
        for item in summary["class_stats"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("LOW CONSISTENCY\n")
        f.write("=" * 80 + "\n")
        for i, rec in enumerate(low_records, start=1):
            f.write(f"[{i}] IMAGE: {rec['image_path']}\n")
            f.write(f"CLASS: {rec['class_name']} | SCORE: {rec['mean_consistency_score']:.4f}\n")
            for pr in rec["prompt_records"]:
                f.write(f"  P{pr['prompt_index']}: {pr['prompt_text']}\n")
                f.write(f"  R{pr['prompt_index']}: {pr['response_text']}\n")
            f.write("-" * 80 + "\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("HIGH CONSISTENCY\n")
        f.write("=" * 80 + "\n")
        for i, rec in enumerate(high_records, start=1):
            f.write(f"[{i}] IMAGE: {rec['image_path']}\n")
            f.write(f"CLASS: {rec['class_name']} | SCORE: {rec['mean_consistency_score']:.4f}\n")
            for pr in rec["prompt_records"]:
                f.write(f"  P{pr['prompt_index']}: {pr['prompt_text']}\n")
                f.write(f"  R{pr['prompt_index']}: {pr['response_text']}\n")
            f.write("-" * 80 + "\n")


def write_summary_csv(records: List[Dict], output_csv_path: Path) -> None:
    """
    Excel/표 작성용 CSV 저장
    """
    rows = []
    for rec in records:
        row = {
            "experiment_name": rec["experiment_name"],
            "timestamp": rec["timestamp"],
            "prompt_group": rec["prompt_group"],
            "image_index": rec["image_index"],
            "image_path": rec["image_path"],
            "class_name": rec["class_name"],
            "mean_consistency_score": rec["mean_consistency_score"],
        }
        for pr in rec["prompt_records"]:
            row[f"prompt_{pr['prompt_index']}"] = pr["prompt_text"]
            row[f"response_{pr['prompt_index']}"] = pr["response_text"]
        rows.append(row)

    fieldnames = sorted({k for row in rows for k in row.keys()})

    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_source_manifest(output_path: Path, gen_cfg: GenerationConfig, sampled_images: List[Path]) -> None:
    """
    source manifest 저장
    - 어떤 소스/문서를 근거로 코드를 구성했는지
    - 어떤 설정으로 실행했는지
    - 어떤 이미지 샘플이 선택되었는지
    """
    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "created_at": datetime.now().isoformat(),
        "generation_config": asdict(gen_cfg),
        "dataset_relative_path": str(DATASET_RELATIVE_PATH),
        "sampled_images_count": len(sampled_images),
        "sampled_images": [str(p) for p in sampled_images],
        "source_manifest": [asdict(x) for x in build_source_manifest()],
        "notes": [
            "본 manifest는 논문/공식문서 작성 시 출처 추적과 재현성 확보를 위한 기록 파일이다.",
            "본 실험은 rephrase prompt set 과 viewpoint prompt set 을 분리하여 수행한다.",
            "이 파일은 원본 전체 복제가 아니라 사용자 제공 코드/로그 기반 재구성 구현임.",
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# =============================================================================
# [9] 메인 실행
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # 1. seed 고정
    # -------------------------------------------------------------------------
    set_seed(SEED)

    # -------------------------------------------------------------------------
    # 2. 출력 경로 준비
    # -------------------------------------------------------------------------
    dirs = prepare_output_dirs()

    # -------------------------------------------------------------------------
    # 3. generation config 구성
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 4. 이미지 샘플링
    # -------------------------------------------------------------------------
    image_paths = sample_images_by_class(SAMPLES_PER_CLASS)

    # -------------------------------------------------------------------------
    # 5. source manifest 저장
    # -------------------------------------------------------------------------
    write_source_manifest(
        output_path=dirs["manifest"] / "source_manifest.json",
        gen_cfg=gen_cfg,
        sampled_images=image_paths,
    )

    # -------------------------------------------------------------------------
    # 6. 모델 로드
    # -------------------------------------------------------------------------
    processor, vlm, sbert = load_models()

    # -------------------------------------------------------------------------
    # 7. 실험 A: rephrase
    # -------------------------------------------------------------------------
    rephrase_jsonl = dirs["jsonl"] / "rephrase_results.jsonl"
    rephrase_records = run_prompt_group_experiment(
        image_paths=image_paths,
        prompt_group_name="rephrase",
        prompts=REPHRASE_PROMPTS,
        processor=processor,
        vlm=vlm,
        sbert=sbert,
        gen_cfg=gen_cfg,
        output_jsonl_path=rephrase_jsonl,
    )

    rephrase_summary = summarize_records(rephrase_records)
    rephrase_low = top_k_records(rephrase_records, k=5, reverse=False)
    rephrase_high = top_k_records(rephrase_records, k=5, reverse=True)

    write_summary_txt(
        prompt_group_name="rephrase",
        summary=rephrase_summary,
        low_records=rephrase_low,
        high_records=rephrase_high,
        output_path=dirs["summary"] / "rephrase_summary.txt",
    )

    write_summary_csv(
        records=rephrase_records,
        output_csv_path=dirs["summary"] / "rephrase_results.csv",
    )

    # -------------------------------------------------------------------------
    # 8. 실험 B: viewpoint
    # -------------------------------------------------------------------------
    viewpoint_jsonl = dirs["jsonl"] / "viewpoint_results.jsonl"
    viewpoint_records = run_prompt_group_experiment(
        image_paths=image_paths,
        prompt_group_name="viewpoint",
        prompts=VIEWPOINT_PROMPTS,
        processor=processor,
        vlm=vlm,
        sbert=sbert,
        gen_cfg=gen_cfg,
        output_jsonl_path=viewpoint_jsonl,
    )

    viewpoint_summary = summarize_records(viewpoint_records)
    viewpoint_low = top_k_records(viewpoint_records, k=5, reverse=False)
    viewpoint_high = top_k_records(viewpoint_records, k=5, reverse=True)

    write_summary_txt(
        prompt_group_name="viewpoint",
        summary=viewpoint_summary,
        low_records=viewpoint_low,
        high_records=viewpoint_high,
        output_path=dirs["summary"] / "viewpoint_summary.txt",
    )

    write_summary_csv(
        records=viewpoint_records,
        output_csv_path=dirs["summary"] / "viewpoint_results.csv",
    )

    # -------------------------------------------------------------------------
    # 9. 최종 비교 리포트 저장
    # -------------------------------------------------------------------------
    final_compare = {
        "experiment_name": EXPERIMENT_NAME,
        "created_at": datetime.now().isoformat(),
        "generation_config": asdict(gen_cfg),
        "rephrase_summary": rephrase_summary,
        "viewpoint_summary": viewpoint_summary,
        "comparison_note": [
            "rephrase와 viewpoint의 평균/분산 차이를 먼저 확인한다.",
            "viewpoint 조건에서 consistency가 더 낮다면, 질문 관점 변화 자체의 영향 가능성을 의심해야 한다.",
            "rephrase 조건에서도 일관되게 낮은 샘플이 있다면, uncertainty candidate로 우선 검토할 수 있다.",
        ],
    }

    with open(dirs["summary"] / "final_compare.json", "w", encoding="utf-8") as f:
        json.dump(final_compare, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[완료] 오늘 실험 종료")
    print(f"출력 경로: {dirs['base']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
