import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import unsloth  # must be imported before trl/transformers/peft for patches
from unsloth import FastLanguageModel


SYSTEM_PROMPT = (
    "你是电商领域的问题意图识别模型。\n"
    "只输出严格 JSON，且必须符合以下结构：\n"
    "{\"labels\":[{\"level1\":\"意图一级名称\",\"level2\":\"意图二级名称\"}]}\n"
    "注意：不要输出省略号或占位符，必须输出真实意图名称。"
)

USER_TEMPLATE = "用户问题：{text}\n请输出意图JSON。"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate intent model with JSON outputs.")
    p.add_argument("--model-dir", required=True, help="Trained model directory.")
    p.add_argument("--test-jsonl", required=True, help="Test data JSONL.")
    p.add_argument("--output-jsonl", default="eval_results.jsonl")
    p.add_argument("--schema-path", default="schema_intent.json")
    p.add_argument("--regex-path", default="regex_intent.txt")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--no-constrained", action="store_true", help="Disable schema constrained decoding.")
    return p


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_regex(path: str) -> re.Pattern:
    with open(path, "r", encoding="utf-8") as f:
        pattern = f.read().strip()
    return re.compile(pattern, re.DOTALL)


def load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_constrained_decode(model, tokenizer, prompt: str, schema: Dict[str, Any], max_new_tokens: int):
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
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        prefix_allowed_tokens_fn=prefix_fn,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def normal_generate(model, tokenizer, prompt: str, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    return text[start : end + 1]


def parse_labels(obj: Dict[str, Any]) -> List[Tuple[str, str]]:
    labels = obj.get("labels", [])
    out = []
    for x in labels:
        l1 = x.get("level1")
        l2 = x.get("level2")
        if l1 and l2:
            out.append((l1, l2))
    return out


def metrics(y_true: List[List[Tuple[str, str]]], y_pred: List[List[Tuple[str, str]]]):
    # Micro
    tp = fp = fn = 0
    all_labels = set()
    for t, p in zip(y_true, y_pred):
        tset, pset = set(t), set(p)
        all_labels |= tset | pset
        tp += len(tset & pset)
        fp += len(pset - tset)
        fn += len(tset - pset)
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    # Macro (per label)
    per_label = {}
    for lbl in all_labels:
        l_tp = l_fp = l_fn = 0
        for t, p in zip(y_true, y_pred):
            tset, pset = set(t), set(p)
            if lbl in tset and lbl in pset:
                l_tp += 1
            elif lbl in pset and lbl not in tset:
                l_fp += 1
            elif lbl in tset and lbl not in pset:
                l_fn += 1
        p = l_tp / (l_tp + l_fp) if (l_tp + l_fp) else 0.0
        r = l_tp / (l_tp + l_fn) if (l_tp + l_fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[lbl] = {"precision": p, "recall": r, "f1": f1}

    if per_label:
        macro_p = sum(v["precision"] for v in per_label.values()) / len(per_label)
        macro_r = sum(v["recall"] for v in per_label.values()) / len(per_label)
        macro_f1 = sum(v["f1"] for v in per_label.values()) / len(per_label)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "per_label": per_label,
    }


def main():
    args = build_arg_parser().parse_args()
    rows = load_jsonl(args.test_jsonl)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    regex = load_regex(args.regex_path)
    schema = load_schema(args.schema_path)

    y_true = []
    y_pred = []

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for r in rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(text=r["text"])},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            raw = None
            if not args.no_constrained:
                raw = try_constrained_decode(
                    model, tokenizer, prompt, schema, args.max_new_tokens
                )
            if raw is None:
                raw = normal_generate(model, tokenizer, prompt, args.max_new_tokens)

            j = extract_json(raw)
            ok = bool(j and regex.match(j))
            pred_labels = []
            if ok:
                try:
                    pred_labels = parse_labels(json.loads(j))
                except Exception:
                    pred_labels = []

            true_labels = [(x["level1"], x["level2"]) for x in r.get("labels", [])]
            y_true.append(true_labels)
            y_pred.append(pred_labels)

            out = {
                "text": r.get("text", ""),
                "true_labels": [{"level1": l1, "level2": l2} for l1, l2 in true_labels],
                "pred_labels": [{"level1": l1, "level2": l2} for l1, l2 in pred_labels],
                "pred_raw": j if ok else "",
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            # Output to stdout for visibility
            print(json.dumps(out, ensure_ascii=False))

    m = metrics(y_true, y_pred)
    print("\n== Metrics ==")
    print(json.dumps(m["micro"], ensure_ascii=False, indent=2))
    print(json.dumps(m["macro"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
