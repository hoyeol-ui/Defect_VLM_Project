import json
import statistics
from datetime import datetime
from pathlib import Path


LOW_K = 5
MID_K = 5
HIGH_K = 5


def get_project_root():
    return Path(__file__).resolve().parent.parent


def find_latest_prompt_ensemble_jsonl():
    root = get_project_root()
    log_dir = root / "logs"

    if not log_dir.exists():
        raise FileNotFoundError(f"❌ logs directory not found: {log_dir}")

    files = sorted(log_dir.glob("prompt_ensemble_*.jsonl"))
    if not files:
        raise FileNotFoundError("❌ No prompt_ensemble_*.jsonl found in logs/")

    return max(files, key=lambda p: p.stat().st_mtime)


def load_records(jsonl_path):
    records = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            if item.get("final_summary", False):
                continue

            if "mean_consistency_score" in item:
                records.append(item)

    if not records:
        raise ValueError("❌ No valid records found.")

    return records


def get_class_name_from_path(image_path):
    # .../train/images/class_name/file.jpg
    return Path(image_path).parent.name


def prepare_output_paths():
    root = get_project_root()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = log_dir / f"prompt_ensemble_analysis_{ts}.txt"
    jsonl_path = log_dir / f"prompt_ensemble_analysis_{ts}.jsonl"
    return txt_path, jsonl_path


def write_txt(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def write_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def summarize_overall(records):
    scores = [r["mean_consistency_score"] for r in records]

    summary = {
        "count": len(scores),
        "mean": round(sum(scores) / len(scores), 4),
        "std": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "range": round(max(scores) - min(scores), 4),
    }
    return summary


def summarize_by_class(records):
    grouped = {}

    for r in records:
        class_name = get_class_name_from_path(r["image_path"])
        grouped.setdefault(class_name, []).append(r["mean_consistency_score"])

    rows = []
    for class_name, scores in grouped.items():
        rows.append({
            "class_name": class_name,
            "count": len(scores),
            "mean": round(sum(scores) / len(scores), 4),
            "std": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        })

    rows = sorted(rows, key=lambda x: x["mean"])
    return rows


def pick_low_mid_high(records):
    sorted_records = sorted(records, key=lambda x: x["mean_consistency_score"])
    n = len(sorted_records)

    low = sorted_records[:LOW_K]
    high = sorted_records[-HIGH_K:]

    mid_center = n // 2
    mid_start = max(0, mid_center - MID_K // 2)
    mid = sorted_records[mid_start:mid_start + MID_K]

    return low, mid, list(reversed(high))


def print_sample_block(title, samples, txt_path, jsonl_path):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    write_txt(txt_path, "=" * 80)
    write_txt(txt_path, title)
    write_txt(txt_path, "=" * 80)

    for idx, item in enumerate(samples, 1):
        image_path = item["image_path"]
        score = item["mean_consistency_score"]
        class_name = get_class_name_from_path(image_path)
        responses = item.get("responses", [])
        pairwise = item.get("pairwise_similarity", [])

        print(f"\n[{idx}] CLASS: {class_name}")
        print(f"IMAGE_PATH: {image_path}")
        print(f"SCORE: {score:.4f}")
        print("RESPONSES:")
        for i, r in enumerate(responses, 1):
            print(f"  [{i}] {r}")
        print("PAIRWISE:")
        for p in pairwise:
            print(f"  ({p['pair'][0]}, {p['pair'][1]}): {p['score']:.4f}")

        write_txt(txt_path, f"[{idx}] CLASS: {class_name}")
        write_txt(txt_path, f"IMAGE_PATH: {image_path}")
        write_txt(txt_path, f"SCORE: {score:.4f}")
        write_txt(txt_path, "RESPONSES:")
        for i, r in enumerate(responses, 1):
            write_txt(txt_path, f"  [{i}] {r}")
        write_txt(txt_path, "PAIRWISE:")
        for p in pairwise:
            write_txt(txt_path, f"  ({p['pair'][0]}, {p['pair'][1]}): {p['score']:.4f}")
        write_txt(txt_path, "-" * 80)

        write_jsonl(jsonl_path, {
            "section": title,
            "class_name": class_name,
            "image_path": image_path,
            "mean_consistency_score": score,
            "responses": responses,
            "pairwise_similarity": pairwise
        })


if __name__ == "__main__":
    source_jsonl = find_latest_prompt_ensemble_jsonl()
    records = load_records(source_jsonl)

    txt_path, jsonl_path = prepare_output_paths()

    overall = summarize_overall(records)
    class_rows = summarize_by_class(records)
    low, mid, high = pick_low_mid_high(records)

    print("SOURCE JSONL:", source_jsonl)
    print("TXT LOG:", txt_path)
    print("JSONL LOG:", jsonl_path)

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    for k, v in overall.items():
        print(f"{k}: {v}")

    write_txt(txt_path, f"SOURCE JSONL: {source_jsonl}")
    write_txt(txt_path, "=" * 80)
    write_txt(txt_path, "OVERALL SUMMARY")
    write_txt(txt_path, "=" * 80)
    for k, v in overall.items():
        write_txt(txt_path, f"{k}: {v}")

    write_jsonl(jsonl_path, {"section": "overall_summary", **overall})

    print("\n" + "=" * 80)
    print("CLASS-WISE SUMMARY")
    print("=" * 80)
    write_txt(txt_path, "=" * 80)
    write_txt(txt_path, "CLASS-WISE SUMMARY")
    write_txt(txt_path, "=" * 80)

    for row in class_rows:
        line = (
            f"{row['class_name']}: "
            f"count={row['count']}, mean={row['mean']}, std={row['std']}, "
            f"min={row['min']}, max={row['max']}"
        )
        print(line)
        write_txt(txt_path, line)
        write_jsonl(jsonl_path, {"section": "class_summary", **row})

    print_sample_block("LOW CONSISTENCY SAMPLES", low, txt_path, jsonl_path)
    print_sample_block("MID CONSISTENCY SAMPLES", mid, txt_path, jsonl_path)
    print_sample_block("HIGH CONSISTENCY SAMPLES", high, txt_path, jsonl_path)

    print("\n" + "=" * 80)
    print("✅ 분석 완료")
    print("=" * 80)

    write_txt(txt_path, "=" * 80)
    write_txt(txt_path, "DONE")
    write_txt(txt_path, "=" * 80)
    write_jsonl(jsonl_path, {"done": True, "source_jsonl": str(source_jsonl)})