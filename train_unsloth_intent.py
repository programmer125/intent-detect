import argparse
import inspect
import json
from typing import List, Tuple

import unsloth  # must be imported before trl/transformers/peft for patches
from unsloth import FastLanguageModel

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

SYSTEM_PROMPT_BASE = (
    "你是电商领域的问题意图识别模型。\n"
    "只输出严格 JSON，且必须符合以下结构：\n"
    "{\"labels\":[{\"level1\":\"意图一级名称\",\"level2\":\"意图二级名称\"}]}\n"
    "注意：不要输出省略号或占位符，必须输出真实意图名称。\n"
    "意图只能从给定列表中选择。"
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
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--intent-map", default="intent_id_map.json")
    return p

# --------- Data ---------

def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_intent_list(path: str) -> Tuple[List[str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    level1 = list(data.get("level1", {}).keys())
    level2 = []
    for k in data.get("level2", {}).keys():
        # key format: "level1/level2"
        parts = k.split("/", 1)
        if len(parts) == 2:
            level2.append(k)
    return level1, level2


def build_system_prompt(intent_map_path: str) -> str:
    level1, level2 = load_intent_list(intent_map_path)
    l1 = "；".join(level1)
    l2 = "；".join(level2)
    return (
        SYSTEM_PROMPT_BASE
        + "\n一级意图列表："
        + l1
        + "\n二级意图列表（格式：一级/二级）："
        + l2
    )


def build_dataset(tokenizer, rows: List[dict], system_prompt: str) -> Dataset:
    data = []
    for r in rows:
        messages = [
            {"role": "system", "content": system_prompt},
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
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    train_rows = load_jsonl(args.train_jsonl)
    val_rows = load_jsonl(args.val_jsonl)

    system_prompt = build_system_prompt(args.intent_map)
    train_ds = build_dataset(tokenizer, train_rows, system_prompt)
    val_ds = build_dataset(tokenizer, val_rows, system_prompt)

    ta_kwargs = dict(
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
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" not in sig.parameters and "eval_strategy" in sig.parameters:
        ta_kwargs["eval_strategy"] = ta_kwargs.pop("evaluation_strategy")
    train_args = TrainingArguments(**ta_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        args=train_args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
