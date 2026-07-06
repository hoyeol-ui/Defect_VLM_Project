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

BATCH_SIZE = 1  # Qwen2-VL은 배치를 1로 시작하는 게 가장 안전함
GRAD_ACCUMULATION_STEPS = 8
EPOCHS = 10
LR = 1e-4


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

        # [수정 1] 인덱싱([0]) 제거하여 processor가 만든 구조 그대로 반환
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")

        return {
            "input_ids": inputs.input_ids,
            "pixel_values": inputs.pixel_values,
            "image_grid_thw": inputs.image_grid_thw,
            "labels": inputs.input_ids.clone()
        }


# [수정 2] 복잡한 collate_fn 삭제 (기본값 사용)

def train():
    print(f"🚀 Qwen2-VL-2B 로딩 중 (Bfloat16 + MPS)...")

    # [수정 3] bfloat16 사용 및 eager 모드 결합
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # [수정 4] LoRA 타겟 모듈 최적화
    config = LoraConfig(
        r=16, lora_alpha=32,
        # 언어와 시각 레이어를 골고루 타겟팅
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", lora_dropout=0.05
    )
    model = get_peft_model(model, config)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jsonl_path = os.path.join(project_root, "data", "active_samples_100.jsonl")

    dataset = ActiveDataset(jsonl_path, processor)
    # [수정 5] collate_fn 없이 기본 로더 사용
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    print(f"\n🔥 학습 시작! (좌표계 보존 모드)")

    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for step, batch in enumerate(pbar):
            # [수정 6] DataLoader가 붙인 불필요한 차원 제거 (.squeeze(0))
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
            loss.backward()

            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUMULATION_STEPS
            pbar.set_postfix(loss=loss.item() * GRAD_ACCUMULATION_STEPS)

        print(f"Epoch {epoch + 1} 완료 - 평균 Loss: {total_loss / len(dataloader):.4f}")

    save_path = os.path.join(project_root, "models", "qwen_lora_adapter")
    model.save_pretrained(save_path)
    print(f"\n✅ 학습 성공! 어댑터 저장 완료.")


if __name__ == "__main__": train()