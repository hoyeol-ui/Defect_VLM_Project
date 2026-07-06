import json
import statistics
from datetime import datetime
from pathlib import Path


LOW_K = 5
MID_K = 5
HIGH_K = 5


# -----------------------------
# 경로
# -----------------------------
def get_project_root():
    return Path(__file__).resolve().parent.parent


def find_latest_prompt_ensemble_jsonl():
    root = get_project_root()
    log_dir = root / "logs"

    files = sorted(log_dir.glob("prompt_ensemble_*.jsonl"))
    if not files:
        raise FileNotFoundError("❌ prompt_ensemble jsonl 없음")

    return max(files, key=lambda p: p.stat().st_mtime)


# -----------------------------
# 로드
# -----------------------------
def load_records(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("final_summary", False):
                continue
            if "mean_consistency_score" in item:
                records.append(item)

    if not records:
        raise ValueError("❌ 유효 데이터 없음")

    return records


# -----------------------------
# 통계
# -----------------------------
def summarize_overall(records):
    scores = [r["mean_consistency_score"] for r in records]

    return {
        "count": len(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "std": round(statistics.pstdev(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "range": round(max(scores) - min(scores), 4),
    }


def get_class_name(path):
    return Path(path).parent.name


def summarize_by_class(records):
    grouped = {}

    for r in records:
        cls = get_class_name(r["image_path"])
        grouped.setdefault(cls, []).append(r["mean_consistency_score"])

    rows = []
    for cls, scores in grouped.items():
        rows.append({
            "class": cls,
            "count": len(scores),
            "mean": round(sum(scores)/len(scores), 4),
            "std": round(statistics.pstdev(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        })

    return sorted(rows, key=lambda x: x["mean"])


# -----------------------------
# 샘플 선택
# -----------------------------
def split_groups(records):
    records = sorted(records, key=lambda x: x["mean_consistency_score"])
    n = len(records)

    low = records[:LOW_K]
    high = records[-HIGH_K:]

    mid_center = n // 2
    mid_start = max(0, mid_center - MID_K // 2)
    mid = records[mid_start:mid_start + MID_K]

    return low, mid, list(reversed(high))


# -----------------------------
# 로그
# -----------------------------
def prepare_output():
    root = get_project_root()
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt = log_dir / f"FULL_analysis_{ts}.txt"
    jsonl = log_dir / f"FULL_analysis_{ts}.jsonl"

    return txt, jsonl


def wtxt(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def wjson(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------
# 출력 함수
# -----------------------------
def print_group(title, group, txt, jsonl):
    print("\n" + "="*80)
    print(title)
    print("="*80)

    wtxt(txt, "="*80)
    wtxt(txt, title)
    wtxt(txt, "="*80)

    for i, r in enumerate(group, 1):
        img = r["image_path"]
        cls = get_class_name(img)
        score = r["mean_consistency_score"]
        responses = r["responses"]
        pairwise = r["pairwise_similarity"]

        print(f"\n[{i}] {img}")
        print(f"CLASS: {cls} | SCORE: {score:.4f}")

        wtxt(txt, f"[{i}] IMAGE: {img}")
        wtxt(txt, f"CLASS: {cls} | SCORE: {score:.4f}")

        for j, res in enumerate(responses, 1):
            print(f"  R{j}: {res}")
            wtxt(txt, f"  R{j}: {res}")

        print("PAIRWISE:")
        for p in pairwise:
            print(f"  ({p['pair'][0]}, {p['pair'][1]}): {p['score']:.4f}")
            wtxt(txt, f"  ({p['pair'][0]}, {p['pair'][1]}): {p['score']:.4f}")

        # 🔥 human inspection 템플릿 포함
        wtxt(txt, "HUMAN REVIEW:")
        wtxt(txt, "- 설명이 맞는가? [Yes / No / Unclear]")
        wtxt(txt, "- 세 설명이 동일한 defect를 의미하는가? [Yes / No / Partly]")
        wtxt(txt, "- 비고:")

        wtxt(txt, "-"*80)

        wjson(jsonl, {
            "section": title,
            "image": img,
            "class": cls,
            "score": score,
            "responses": responses,
            "pairwise": pairwise
        })


# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    source = find_latest_prompt_ensemble_jsonl()
    records = load_records(source)

    txt, jsonl = prepare_output()

    print("SOURCE:", source)
    print("TXT:", txt)
    print("JSONL:", jsonl)

    overall = summarize_overall(records)
    class_stats = summarize_by_class(records)

    # 전체 통계
    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(k, v)
        wtxt(txt, f"{k}: {v}")

    wjson(jsonl, {"overall": overall})

    # 클래스별
    print("\n=== CLASS ===")
    for row in class_stats:
        print(row)
        wtxt(txt, str(row))
        wjson(jsonl, {"class": row})

    # 그룹
    low, mid, high = split_groups(records)

    print_group("LOW CONSISTENCY", low, txt, jsonl)
    print_group("MID CONSISTENCY", mid, txt, jsonl)
    print_group("HIGH CONSISTENCY", high, txt, jsonl)

    print("\n✅ FULL 분석 완료")
    wtxt(txt, "DONE")
    wjson(jsonl, {"done": True})