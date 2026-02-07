import json
import os
import re
from typing import Any, Dict

from unsloth import FastLanguageModel

MODEL_DIR = os.environ.get("MODEL_DIR", "output/qwen2p5_intent_qlora")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))
SCHEMA_PATH = os.environ.get("SCHEMA_PATH", "schema_intent.json")
REGEX_PATH = os.environ.get("REGEX_PATH", "regex_intent.txt")

SYSTEM_PROMPT = (
    "你是电商领域的问题意图识别模型。\n"
    "只输出严格 JSON，且必须符合 schema："
    "{\"labels\":[{\"level1\":\"...\",\"level2\":\"...\"}]}"
)

USER_TEMPLATE = "用户问题：{text}\n请输出意图JSON。"


def load_regex() -> re.Pattern:
    with open(REGEX_PATH, "r", encoding="utf-8") as f:
        pattern = f.read().strip()
    return re.compile(pattern, re.DOTALL)


def load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def try_constrained_decode(model, tokenizer, prompt: str):
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

    schema = load_schema()
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


def infer(text: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    regex = load_regex()

    # 1) try constrained decode (if available)
    out = try_constrained_decode(model, tokenizer, prompt)
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
                f"用户问题：{text}"
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
    # quick manual test
    print(infer("快递用哪家？下单后多久发货？"))
