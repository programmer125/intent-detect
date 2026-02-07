**Unsloth QLoRA 训练与推理（严格 JSON）**

**目标**
- 基座：`Qwen2.5-7B-Instruct`
- 训练：QLoRA（4bit）
- 输出：严格 JSON（schema + regex 校验 + 约束解码）

**文件**
- 训练脚本：`train_unsloth_intent.py`
- 推理脚本：`infer_unsloth_intent.py`
- JSON Schema：`schema_intent.json`
- 正则校验：`regex_intent.txt`

**训练数据**
- `data/intent_train_1k.train.jsonl`
- `data/intent_train_1k.val.jsonl`

**训练命令**
```bash
python3 train_unsloth_intent.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-jsonl data/intent_train_1k.train.jsonl \
  --val-jsonl data/intent_train_1k.val.jsonl \
  --output-dir output/qwen2p5_intent_qlora
```

**推理命令**
```bash
export MODEL_DIR=output/qwen2p5_intent_qlora
python3 infer_unsloth_intent.py
```

**约束解码说明**
- 脚本会尝试使用 `lm-format-enforcer` 做 JSON schema 约束解码
- 如果未安装该依赖，会自动退化为普通解码 + regex 校验 + 失败重试

**建议的下一步**
- 扩充真实样本，提升领域覆盖
- 在 `intent_hard_negatives.jsonl` 上做混淆评估
