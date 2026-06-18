"""
步骤5（可选）：导出 ONNX 模型 + 推理加速对比
- 将 PyTorch 模型转换为 ONNX 格式
- ONNX Runtime 推理，速度提升 2-5x
- 对比原始 vs ONNX 推理速度
"""
import os
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ⚠️ 必须在导入前设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.makedirs('models/onnx', exist_ok=True)

# ── 1. 选择源模型 ───────────────────────────────
# 优先使用 roberta-finetuned，其次 bert-finetuned
model_path = None
for p in ['./models/roberta-finetuned', './models/bert-finetuned']:
    if os.path.exists(p):
        model_path = p
        break

if model_path is None:
    print("❌ 未找到已训练的模型，请先运行 03_train_bert.py")
    exit(1)

print("=" * 60)
print("         ONNX 模型导出 & 推理加速")
print("=" * 60)
print(f"  源模型: {model_path}")

# ── 2. 加载 PyTorch 模型 ────────────────────────
print("\n[1/4] 加载 PyTorch 模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"  设备: {device}")

# ── 3. 导出 ONNX ────────────────────────────────
print("\n[2/4] 导出 ONNX...")
onnx_path = "./models/onnx/sentiment_model.onnx"

# 准备示例输入（动态 batch + 固定长度）
sample_text = "这个商品质量很好"
inputs = tokenizer(sample_text, return_tensors='pt', padding='max_length',
                    max_length=256, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 动态轴配置
dynamic_axes = {
    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
    'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
    'logits': {0: 'batch_size'},
}

# 导出
with torch.no_grad():
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']),
        onnx_path,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

import subprocess
result = subprocess.run(['du', '-h', onnx_path.replace('./', '')],
                        capture_output=True, text=True, shell=True)
size_str = result.stdout.split()[0] if result.returncode == 0 else '?'
print(f"  ✅ 已导出: {onnx_path} ({size_str})")

# ── 4. 速度对比 ─────────────────────────────────
print("\n[3/4] 推理速度对比...")

test_texts = [
    "这个商品质量很好，物流也快，非常满意",
    "太差了，用了两天就坏了",
    "还行吧，一般般",
    "快递很快但包装简陋",
    "味道不错，价格实惠，推荐购买",
] * 20  # 100 条

# 预热
for _ in range(3):
    _ = tokenizer(test_texts[0], return_tensors='pt', truncation=True, max_length=256)

# PyTorch 推理
def benchmark_pytorch(texts, use_gpu=True):
    model.eval()
    times = []
    for text in texts:
        t0 = time.time()
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        times.append((time.time() - t0) * 1000)
    return np.mean(times), np.std(times)

# ONNX Runtime 推理
def benchmark_onnx(texts):
    try:
        import onnxruntime
    except ImportError:
        print("  ⚠️ onnxruntime 未安装，跳过 ONNX 测试")
        print("    安装: pip install onnxruntime")
        return None, None

    session = onnxruntime.InferenceSession(onnx_path)
    times = []
    for text in texts:
        t0 = time.time()
        inputs = tokenizer(text, return_tensors='np', truncation=True,
                           max_length=256, padding='max_length')
        ort_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'token_type_ids': inputs['token_type_ids'],
        }
        logits = session.run(None, ort_inputs)[0]
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        times.append((time.time() - t0) * 1000)
    return np.mean(times), np.std(times)

print("  测试中...")
pt_mean, pt_std = benchmark_pytorch(test_texts)
print(f"  PyTorch:   {pt_mean:.1f} ± {pt_std:.1f} ms/条")

onnx_mean, onnx_std = benchmark_onnx(test_texts)
if onnx_mean is not None:
    print(f"  ONNX RT:   {onnx_mean:.1f} ± {onnx_std:.1f} ms/条")
    speedup = pt_mean / onnx_mean
    print(f"  加速比:    {speedup:.1f}x")
else:
    print("  ONNX RT:   未测试（需 pip install onnxruntime）")

# ── 5. ONNX 推理函数（供 app.py 调用）────────────
print("\n[4/4] 生成 ONNX 推理模块...")

onnx_infer_code = '''
"""ONNX 推理封装 — 比 PyTorch pipeline 快 2-5x"""
import onnxruntime
import numpy as np

class ONNXSentiment:
    def __init__(self, onnx_path="./models/onnx/sentiment_model.onnx",
                 tokenizer_path=None):
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        from transformers import AutoTokenizer

        self.session = onnxruntime.InferenceSession(onnx_path)
        # 自动找对应的 tokenizer
        if tokenizer_path is None:
            for p in ['./models/roberta-finetuned', './models/bert-finetuned']:
                if os.path.exists(p):
                    tokenizer_path = p
                    break
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = 256

    def predict(self, text):
        """单条预测，返回 (label, confidence)"""
        inputs = self.tokenizer(
            text, return_tensors='np', truncation=True,
            max_length=self.max_length, padding='max_length',
        )
        ort_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'token_type_ids': inputs['token_type_ids'],
        }
        logits = self.session.run(None, ort_inputs)[0]
        probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        pred = int(np.argmax(probs[0]))
        conf = float(probs[0][pred])
        return pred, conf

    def predict_batch(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

# 使用示例:
# model = ONNXSentiment()
# label, confidence = model.predict("这个商品很好")
'''

with open('models/onnx/onnx_infer.py', 'w', encoding='utf-8') as f:
    f.write(onnx_infer_code.strip())

print("  ✅ 已生成: models/onnx/onnx_infer.py")

print("\n" + "=" * 60)
print("         ONNX 导出完成！")
print("=" * 60)
print(f"""
  📁 产出文件:
     models/onnx/sentiment_model.onnx   — ONNX 模型
     models/onnx/onnx_infer.py           — 推理封装类

  🚀 使用方法:
     from models.onnx.onnx_infer import ONNXSentiment
     model = ONNXSentiment()
     label, conf = model.predict("输入文本")
""")