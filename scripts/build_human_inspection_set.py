import json
from datetime import datetime
from pathlib import Path


N_PER_GROUP = 3


def get_project_root():
    return Path(__file__).resolve().parent.parent


def find_latest_prompt_ensemble_jsonl():
    root = get_project_root()
    log_dir = root / "logs"

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


def prepare_output_paths():
    root = get_project_root()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = log_dir / f"human_inspection_set_{ts}.txt"
    jsonl_path = log_dir / f"human_inspection_set_{ts}.jsonl"
    return txt_path, jsonl_path


def write_txt(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def write_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_class_name(image_path):
    return Path(image_path).parent.name


def pick_groups(records):
    records = sorted(records, key=lambda x: x["mean_consistency_score"])
    n = len(records)

    low = records[:N_PER_GROUP]
    high = records[-N_PER_GROUP:]

    mid_center = n // 2
    mid_start = max(0, mid_center - N_PER_GROUP // 2)
    mid = records[mid_start:mid_start + N_PER_GROUP]

    return low, mid, list(reversed(high))


def write_group(title, group, txt_path, jsonl_path):
    write_txt(txt_path, "=" * 80)
    write_txt(txt_path, title)
    write_txt(txt_path, "=" * 80)

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for idx, item in enumerate(group, 1):
        image_path = item["image_path"]
        class_name = get_class_name(image_path)
        score = item["mean_consistency_score"]
        responses = item.get("responses", [])

        print(f"\n[{idx}] {image_path}")
        print(f"CLASS: {class_name}")
        print(f"SCORE: {score:.4f}")
        for i, r in enumerate(responses, 1):
            print(f"  RESPONSE {i}: {r}")

        write_txt(txt_path, f"[{idx}] IMAGE_PATH: {image_path}")
        write_txt(txt_path, f"CLASS: {class_name}")
        write_txt(txt_path, f"SCORE: {score:.4f}")
        for i, r in enumerate(responses, 1):
            write_txt(txt_path, f"RESPONSE {i}: {r}")

        write_txt(txt_path, "HUMAN REVIEW TEMPLATE:")
        write_txt(txt_path, "- 설명이 대체로 맞는가? [Yes / No / Unclear]")
        write_txt(txt_path, "- 세 설명이 같은 defect를 말하는가? [Yes / No / Partly]")
        write_txt(txt_path, "- 비고:")
        write_txt(txt_path, "-" * 80)

        write_jsonl(jsonl_path, {
            "section": title,
            "image_path": image_path,
            "class_name": class_name,
            "mean_consistency_score": score,
            "responses": responses,
            "human_review": {
                "description_correct": None,
                "responses_consistent": None,
                "note": ""
            }
        })


if __name__ == "__main__":
    source_jsonl = find_latest_prompt_ensemble_jsonl()
    records = load_records(source_jsonl)
    txt_path, jsonl_path = prepare_output_paths()

    low, mid, high = pick_groups(records)

    print("SOURCE JSONL:", source_jsonl)
    print("TXT LOG:", txt_path)
    print("JSONL LOG:", jsonl_path)

    write_txt(txt_path, f"SOURCE JSONL: {source_jsonl}")
    write_txt(txt_path, f"N_PER_GROUP: {N_PER_GROUP}")

    write_group("LOW CONSISTENCY REVIEW SET", low, txt_path, jsonl_path)
    write_group("MID CONSISTENCY REVIEW SET", mid, txt_path, jsonl_path)
    write_group("HIGH CONSISTENCY REVIEW SET", high, txt_path, jsonl_path)

    print("\n" + "=" * 80)
    print("✅ human inspection set 생성 완료")
    print("=" * 80)

    write_txt(txt_path, "=" * 80)
    write_txt(txt_path, "DONE")
    write_txt(txt_path, "=" * 80)
    write_jsonl(jsonl_path, {"done": True, "source_jsonl": str(source_jsonl)})