import json
from datetime import datetime
from pathlib import Path


TOP_K = 5


def get_project_root():
    return Path(__file__).resolve().parent.parent


def find_latest_prompt_ensemble_jsonl():
    root = get_project_root()
    log_dir = root / "logs"

    if not log_dir.exists():
        raise FileNotFoundError(f"❌ logs directory not found: {log_dir}")

    jsonl_files = sorted(log_dir.glob("prompt_ensemble_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError("❌ No prompt_ensemble_*.jsonl files found in logs/")

    latest_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
    return latest_file


def load_prompt_ensemble_records(jsonl_path):
    records = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            # final summary line 제외
            if item.get("final_summary", False):
                continue

            if "mean_consistency_score" in item:
                records.append(item)

    if not records:
        raise ValueError("❌ No valid sample records found in jsonl file.")

    return records


def prepare_output_paths():
    root = get_project_root()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = log_dir / f"low_consistency_top{TOP_K}_{ts}.txt"
    jsonl_path = log_dir / f"low_consistency_top{TOP_K}_{ts}.jsonl"

    return txt_path, jsonl_path


def write_txt(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def write_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    latest_jsonl = find_latest_prompt_ensemble_jsonl()
    records = load_prompt_ensemble_records(latest_jsonl)

    sorted_records = sorted(records, key=lambda x: x["mean_consistency_score"])
    bottom_k = sorted_records[:TOP_K]

    txt_out, jsonl_out = prepare_output_paths()

    print("SOURCE JSONL:", latest_jsonl)
    print("TOTAL RECORDS:", len(records))
    print("TOP_K:", TOP_K)
    print("TXT LOG:", txt_out)
    print("JSONL LOG:", jsonl_out)

    write_txt(txt_out, f"SOURCE JSONL: {latest_jsonl}")
    write_txt(txt_out, f"TOTAL RECORDS: {len(records)}")
    write_txt(txt_out, f"TOP_K: {TOP_K}")
    write_txt(txt_out, "=" * 80)

    print("\n" + "=" * 80)
    print(f"LOWEST {TOP_K} CONSISTENCY SAMPLES")
    print("=" * 80)

    for rank, item in enumerate(bottom_k, 1):
        image_path = item["image_path"]
        score = item["mean_consistency_score"]
        responses = item.get("responses", [])
        pairwise = item.get("pairwise_similarity", [])
        prompts = item.get("prompts", [])

        print(f"\n[{rank}] SCORE: {score:.4f}")
        print(f"IMAGE_PATH: {image_path}")

        if prompts:
            print("PROMPT + RESPONSE")
            for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
                print(f"  [{i}] PROMPT: {prompt}")
                print(f"      RESPONSE: {response}")
        else:
            print("RESPONSES")
            for i, response in enumerate(responses, 1):
                print(f"  [{i}] {response}")

        if pairwise:
            print("PAIRWISE SIMILARITY")
            for pair_item in pairwise:
                pair = pair_item["pair"]
                sim = pair_item["score"]
                print(f"  ({pair[0]}, {pair[1]}): {sim:.4f}")

        write_txt(txt_out, f"[{rank}] SCORE: {score:.4f}")
        write_txt(txt_out, f"IMAGE_PATH: {image_path}")

        if prompts:
            write_txt(txt_out, "PROMPT + RESPONSE")
            for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
                write_txt(txt_out, f"  [{i}] PROMPT: {prompt}")
                write_txt(txt_out, f"      RESPONSE: {response}")
        else:
            write_txt(txt_out, "RESPONSES")
            for i, response in enumerate(responses, 1):
                write_txt(txt_out, f"  [{i}] {response}")

        if pairwise:
            write_txt(txt_out, "PAIRWISE SIMILARITY")
            for pair_item in pairwise:
                pair = pair_item["pair"]
                sim = pair_item["score"]
                write_txt(txt_out, f"  ({pair[0]}, {pair[1]}): {sim:.4f}")

        write_txt(txt_out, "-" * 80)

        write_jsonl(jsonl_out, {
            "rank": rank,
            "image_path": image_path,
            "mean_consistency_score": score,
            "responses": responses,
            "pairwise_similarity": pairwise,
            "prompts": prompts
        })

    print("\n" + "=" * 80)
    print("✅ low-consistency 샘플 추출 완료")
    print("=" * 80)

    write_txt(txt_out, "=" * 80)
    write_txt(txt_out, "DONE")
    write_txt(txt_out, "=" * 80)

    write_jsonl(jsonl_out, {
        "done": True,
        "source_jsonl": str(latest_jsonl),
        "top_k": TOP_K
    })