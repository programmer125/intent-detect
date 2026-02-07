import argparse
import json
from typing import List

import unsloth  # must be imported before trl/transformers/peft for patches
from unsloth import FastLanguageModel

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

SYSTEM_PROMPT = (
    "你是电商领域的问题意图识别模型。\n"
    "只输出严格 JSON，且必须符合 schema："
    "{\"labels\":[{\"level1\":\"...\",\"level2\":\"...\"}]}"
)

USER_TEMPLATE = "用户问题：{text}\n请输出意图JSON。"

# --------- Args ---------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unsloth QLoRA training for intent classification.")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train-jsonl", default="data/intent_train_1k.train.jsonl")
    p.add_argument("--val-jsonl", default="data/intent_train_1k.val.jsonl")
    p.add_argument("--output-dir", default="output/qwen2p5_intent_qlora")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)
    return p

# --------- Data ---------

def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(tokenizer, rows: List[dict]) -> Dataset:
    data = []
    for r in rows:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=r["text"])},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        answer = json.dumps({"labels": r["labels"]}, ensure_ascii=False)
        data.append({"text": prompt + answer})
    return Dataset.from_list(data)


# --------- Train ---------

def main():
    args = build_arg_parser().parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    train_rows = load_jsonl(args.train_jsonl)
    val_rows = load_jsonl(args.val_jsonl)

    train_ds = build_dataset(tokenizer, train_rows)
    val_ds = build_dataset(tokenizer, val_rows)

    args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        args=args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
