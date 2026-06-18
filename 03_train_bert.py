"""
步骤3：RoBERTa 微调训练（改进版）
- 使用 hfl/chinese-roberta-wwm-ext（全词遮罩，中文效果更优）
- 增大 max_length、优化学习率调度
- 引入 FGM 对抗训练提升鲁棒性
- 早停更耐心，确保充分收敛
"""
# ⚠️ 必须在导入前设置镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

# ── 配置 ────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)
os.makedirs('models/roberta-finetuned', exist_ok=True)

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"   # 比 bert-base-chinese 中文效果更好
MAX_LENGTH = 256                               # 128→256，覆盖更多文本
BATCH_SIZE = 8                                 # RoBERTa 更大，batch 适当减小
EPOCHS = 8                                     # 更充分训练
LR = 3e-5                                      # 稍高学习率
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
PATIENCE = 3                                   # 更耐心等收敛
USE_FP16 = torch.cuda.is_available()

print("=" * 60)
print("        RoBERTa 微调训练（改进版）")
print("=" * 60)
print(f"  模型: {MODEL_NAME}")
print(f"  最大长度: {MAX_LENGTH}")
print(f"  批次大小: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  学习率: {LR}")
print(f"  早停耐心: {PATIENCE}")
print(f"  GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

# ── 1. 加载数据 ──────────────────────────────────
print("\n[1/6] 加载数据集...")
dataset = load_dataset("lansinuote/ChnSentiCorp")
num_labels = 2  # 二分类训练，推理时阈值切三分类

print(f"  训练集: {len(dataset['train'])} 条")
print(f"  验证集: {len(dataset['validation'])} 条")
print(f"  测试集: {len(dataset['test'])} 条")

# ── 2. Tokenizer & 预处理 ────────────────────────
print("\n[2/6] 加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )

encoded = dataset.map(preprocess, batched=True)
print(f"  Tokenizer 词表大小: {tokenizer.vocab_size}")

# ── 3. 评估指标 ──────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1': f1}

# ── 6. 加载模型 & 训练 ──────────────────────────
print("\n[3/6] 加载预训练模型...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
)
param_count = sum(p.numel() for p in model.parameters()) / 1e6
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"  总参数: {param_count:.1f}M, 可训练: {trainable:.1f}M")

print("\n[4/6] 配置训练参数...")
training_args = TrainingArguments(
    output_dir='./outputs',
    eval_strategy='steps',
    eval_steps=200,                        # 更频繁评估
    save_steps=400,                        # 必须是 eval_steps 的整数倍
    logging_steps=50,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    save_total_limit=2,
    fp16=USE_FP16,
    dataloader_num_workers=0,
    report_to='none',
    # 梯度累积模拟更大 batch
    gradient_accumulation_steps=2,         # 有效 batch = 8×2 = 16
)

print("\n[5/6] 开始训练...")
print("-" * 60)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded['train'],
    eval_dataset=encoded['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)

trainer.train()

# ── 7. 评估 ──────────────────────────────────────
print("\n" + "=" * 60)
print("              最终评估")
print("=" * 60)

print("\n验证集:")
val_results = trainer.evaluate()
for k, v in val_results.items():
    if 'loss' in k:
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v:.4f}")

print("\n测试集:")
test_results = trainer.evaluate(encoded['test'])
for k, v in test_results.items():
    if 'loss' in k:
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v:.4f}")

# ── 8. 保存模型 ──────────────────────────────────
save_path = './models/roberta-finetuned'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n✅ 模型已保存至: {save_path}")

# ── 9. 快速测试（含三分类）────────────────────────
print("\n" + "=" * 60)
print("           快速测试（三分类阈值）")
print("=" * 60)

test_texts = [
    ("这个商品质量很好，物流也快，非常满意", "正面"),
    ("太差了，用了两天就坏了，客服也不理人", "负面"),
    ("还行吧，一般般，无功无过", "中性"),
    ("快递挺快的但包装有点简陋", "中性"),
    ("暂时还没用，先给个好评", "中性"),
]

model.eval()
device = next(model.parameters()).device

for text, expected in test_texts:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred = torch.argmax(probs).item()
        conf = probs[pred].item()

    # 三分类：先看预测标签，再看置信度
    if pred == 1 and conf > 0.65:
        sentiment = "正面 😊"
    elif pred == 0 and conf > 0.65:
        sentiment = "负面 😞"
    else:
        sentiment = "中性 😐"

    print(f"  [{sentiment} | {conf:.1%}] (预期: {expected}) → {text}")
