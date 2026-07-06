import sys, os, json, torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# --- [1] 설정 ---
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 8
EPOCHS = 10
LR = 5e-5  # [수정] 학습률을 낮춰서(1e-4 -> 5e-5) 로스가 튀는 걸 방지


# --- [2] 데이터셋 ---
class ActiveDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = [
            {"role": "user",
             "content": [{"type": "image", "image": item['image']}, {"type": "text", "text": "Describe the defect."}]},
            {"role": "assistant", "content": [{"type": "text", "text": item['suffix']}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")

        return {
            "input_ids": inputs.input_ids,
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw,
            "labels": inputs.input_ids.clone()
        }


def train():
    print(f"🚀 Qwen2-VL-2B 로딩 중 (Stability Mode)...")

    # [수정 1] float32 사용: bfloat16에서 nan이 뜨면 가장 확실한 해결책은 float32로 돌아가는 것임
    # 메모리는 더 먹지만 2B 모델에 배치가 1이라 맥북에서 충분히 버팀
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map=DEVICE,
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM", lora_dropout=0.05
    )
    model = get_peft_model(model, config)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jsonl_path = os.path.join(project_root, "data", "active_samples_100.jsonl")

    dataset = ActiveDataset(jsonl_path, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    print(f"\n🔥 학습 시작! (로스 폭주 방지 장치 가동)")

    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].squeeze(0).to(DEVICE)
            pixel_values = batch["pixel_values"].squeeze(0).to(DEVICE)
            image_grid_thw = batch["image_grid_thw"].squeeze(0).to(DEVICE)
            labels = batch["labels"].squeeze(0).to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels
            )

            loss = outputs.loss / GRAD_ACCUMULATION_STEPS

            # [수정 2] 로스가 nan인지 체크
            if torch.isnan(loss):
                print(f"⚠️ Warning: NaN loss detected at step {step}. Skipping...")
                continue

            loss.backward()

            # [수정 3] 그래디언트 클리핑: 숫자가 너무 커지지 않게 강제로 깎음
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUMULATION_STEPS
            pbar.set_postfix(loss=loss.item() * GRAD_ACCUMULATION_STEPS)

        print(f"Epoch {epoch + 1} 완료 - 평균 Loss: {total_loss / len(dataloader):.4f}")

    save_path = os.path.join(project_root, "models", "qwen_lora_adapter")
    model.save_pretrained(save_path)
    print(f"\n✅ 학습 성공! 이제 숫자가 튀지 않습니다.")


if __name__ == "__main__": train()