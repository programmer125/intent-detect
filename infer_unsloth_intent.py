import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

from unsloth import FastLanguageModel

SYSTEM_PROMPT_BASE = (
    "你是电商领域的问题意图识别模型。\n"
    "只输出严格 JSON。\n"
    "输出键名固定为 level1 与 level2，且其取值必须为真实意图名称。\n"
    "不要输出示例文本、占位符或解释。\n"
    "意图只能从给定列表中选择。"
)

USER_TEMPLATE = "用户问题：{text}\n请输出意图JSON。"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference for intent model with JSON outputs.")
    p.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "output/qwen2p5_intent_qlora"))
    p.add_argument("--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "512")))
    p.add_argument("--schema-path", default=os.environ.get("SCHEMA_PATH", "schema_intent.json"))
    p.add_argument("--regex-path", default=os.environ.get("REGEX_PATH", "regex_intent.txt"))
    p.add_argument("--intent-map", default="intent_id_map.json")
    p.add_argument("--text", default="快递用哪家？下单后多久发货？")
    return p


def load_regex(path: str) -> re.Pattern:
    with open(path, "r", encoding="utf-8") as f:
        pattern = f.read().strip()
    return re.compile(pattern, re.DOTALL)


def load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_intent_list(path: str) -> Tuple[List[str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    level1 = list(data.get("level1", {}).keys())
    level2 = []
    for k in data.get("level2", {}).keys():
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


def try_constrained_decode(model, tokenizer, prompt: str, schema: Dict[str, Any]):
    """
    Optional: If lm-format-enforcer is installed, use it for schema-based decoding.
    Otherwise return None to fall back.
    """
    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
    except Exception:
        return None

    parser = JsonSchemaParser(schema)
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        prefix_allowed_tokens_fn=prefix_fn,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def normal_generate(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_json(text: str) -> str:
    # try to extract first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    return text[start : end + 1]


def validate_json(s: str, regex: re.Pattern) -> bool:
    if not s:
        return False
    return bool(regex.match(s))


def infer(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    system_prompt = build_system_prompt(args.intent_map)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_TEMPLATE.format(text=args.text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    regex = load_regex(args.regex_path)
    schema = load_schema(args.schema_path)

    # 1) try constrained decode (if available)
    out = try_constrained_decode(model, tokenizer, prompt, schema)
    if out is None:
        out = normal_generate(model, tokenizer, prompt)

    j = extract_json(out)
    if validate_json(j, regex):
        return j

    # 2) one retry with explicit repair prompt
    repair_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "你刚才输出的JSON不合法。请只输出合法JSON，不要任何解释。\n"
                f"用户问题：{args.text}"
            ),
        },
    ]
    repair_prompt = tokenizer.apply_chat_template(
        repair_messages, tokenize=False, add_generation_prompt=True
    )
    out2 = normal_generate(model, tokenizer, repair_prompt)
    j2 = extract_json(out2)
    if validate_json(j2, regex):
        return j2

    # 3) fallback: empty
    return "{}"


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    print(infer(args))
