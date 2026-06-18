"""
步骤4：模型评估 & 对比（改进版）
- ML 基线 vs BERT vs RoBERTa
- 二分类 + 三分类（阈值法）指标
- 混淆矩阵 + 错误分析
"""
# ⚠️ 必须在导入前设置镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import pickle
import jieba
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

os.makedirs('models', exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("       模型评估 & 对比（改进版）")
print("=" * 60)

# ── 1. 加载测试集 ───────────────────────────────
print("\n[1/5] 加载测试集...")
dataset = load_dataset("lansinuote/ChnSentiCorp")
X_test = list(dataset['test']['text'])
y_test = list(dataset['test']['label'])
print(f"  测试样本: {len(X_test)}")

# ── 2. 加载模型 ──────────────────────────────────
models = {}

# ML 基线
try:
    with open('models/ml_model.pkl', 'rb') as f:
        ml_clf = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    models['ML (LinearSVM)'] = {
        'type': 'ml',
        'model': ml_clf,
        'vectorizer': tfidf,
    }
    print("  ✅ ML 模型加载成功")
except FileNotFoundError:
    print("  ⚠️ ML 模型未找到 (运行 02_baseline_ml.py)")

# BERT
try:
    bp = './models/bert-finetuned'
    tokenizer = AutoTokenizer.from_pretrained(bp)
    model = AutoModelForSequenceClassification.from_pretrained(bp)
    models['BERT-base'] = {
        'type': 'bert',
        'pipe': pipeline("text-classification", model=model, tokenizer=tokenizer,
                          device=-1, truncation=True, max_length=512),
    }
    print("  ✅ BERT 模型加载成功")
except Exception:
    print("  ⚠️ BERT 模型未找到")

# RoBERTa
try:
    rp = './models/roberta-finetuned'
    tokenizer = AutoTokenizer.from_pretrained(rp)
    model = AutoModelForSequenceClassification.from_pretrained(rp)
    models['RoBERTa-wwm'] = {
        'type': 'bert',
        'pipe': pipeline("text-classification", model=model, tokenizer=tokenizer,
                          device=-1, truncation=True, max_length=512),
    }
    print("  ✅ RoBERTa 模型加载成功")
except Exception:
    print("  ⚠️ RoBERTa 模型未找到 (运行 03_train_bert.py)")

# ── 3. 预测 ──────────────────────────────────────
def predict_ml(texts, model_info):
    cut_texts = [' '.join(jieba.cut(t)) for t in texts]
    vec = model_info['vectorizer'].transform(cut_texts)
    preds = model_info['model'].predict(vec)
    if hasattr(model_info['model'], 'predict_proba'):
        probas = model_info['model'].predict_proba(vec)
    else:
        # SVM: decision_function → sigmoid 伪概率
        d = model_info['model'].decision_function(vec)
        probas = np.zeros((len(d), 2))
        probas[:, 1] = 1.0 / (1.0 + np.exp(-d))     # class 1 概率
        probas[:, 0] = 1.0 - probas[:, 1]            # class 0 概率
    scores = [float(probas[i][p]) for i, p in enumerate(preds)]
    return preds, scores

def predict_bert(texts, model_info):
    pipe = model_info['pipe']
    results = pipe(texts, batch_size=32)
    preds, scores = [], []
    for r in results:
        label = r['label']
        if isinstance(label, int):
            p = label
        elif 'LABEL_1' in str(label) or label == '1':
            p = 1
        else:
            p = 0
        preds.append(p)
        scores.append(r['score'])
    return preds, scores

all_results = {}

for name, info in models.items():
    print(f"\n[{'ML' if info['type']=='ml' else 'NN'}] 评估 {name}...")
    t0 = time.time()

    if info['type'] == 'ml':
        preds, scores = predict_ml(X_test, info)
    else:
        preds, scores = predict_bert(X_test, info)

    elapsed = (time.time() - t0) / len(X_test) * 1000

    # 二分类指标
    bin_acc = accuracy_score(y_test, preds)
    bin_f1 = f1_score(y_test, preds, average='macro')

    # 三分类（阈值法）: 先看预测标签，再看置信度
    ternary_preds = []
    for s, p in zip(scores, preds):
        if p == 1 and s > 0.65:
            ternary_preds.append(2)      # 正面
        elif p == 0 and s > 0.65:
            ternary_preds.append(0)      # 负面
        else:
            ternary_preds.append(1)      # 中性

    # 真实标签映射到三分类（原数据只有正/负，中性按置信度近似）
    # 注：由于 ChnSentiCorp 没有中性标签，这里展示阈值分布统计
    neu_count = sum(1 for p in ternary_preds if p == 1)
    pos_count = sum(1 for p in ternary_preds if p == 2)
    neg_count = sum(1 for p in ternary_preds if p == 0)

    all_results[name] = {
        'binary_acc': bin_acc,
        'binary_f1': bin_f1,
        'ternary_dist': (pos_count, neu_count, neg_count),
        'preds': preds,
        'scores': scores,
        'elapsed': elapsed,
    }

    print(f"  二分类准确率: {bin_acc:.4f} ({bin_acc*100:.2f}%)")
    print(f"  二分类 F1:    {bin_f1:.4f}")
    print(f"  三分类分布:   正面={pos_count} | 中性={neu_count} | 负面={neg_count}")
    print(f"  推理速度:     {elapsed:.1f}ms/条")

# ── 4. 对比汇总 ──────────────────────────────────
print("\n" + "=" * 60)
print("                对比汇总")
print("=" * 60)
print(f"\n{'模型':<20} {'二分类 Acc':<12} {'二分类 F1':<12} {'推理速度':<12} {'三分类分布':<20}")
print("-" * 80)

for name, r in all_results.items():
    pos, neu, neg = r['ternary_dist']
    dist = f"正{pos}/中{neu}/负{neg}"
    speed_str = f"{r['elapsed']:.1f}ms" if r['elapsed'] >= 0.1 else "<0.1ms"
    print(f"{name:<20} {r['binary_acc']:<12.4f} {r['binary_f1']:<12.4f} {speed_str:<12} {dist:<20}")

# ── 5. 混淆矩阵 ──────────────────────────────────
if len(all_results) >= 2:
    print("\n[4/5] 绘制混淆矩阵...")
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5*n+2, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, all_results.items()):
        cm = confusion_matrix(y_test, r['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['负面', '正面'],
                    yticklabels=['负面', '正面'],
                    annot_kws={'fontsize': 14})
        ax.set_title(f'{name}\nAcc: {r["binary_acc"]:.2%}', fontsize=13, fontweight='bold')
        ax.set_xlabel('预测')
        ax.set_ylabel('真实')

    plt.tight_layout()
    cm_path = 'models/model_comparison_v2.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"  📊 混淆矩阵已保存至: {cm_path}")
    plt.close()

# ── 6. 错误分析 ──────────────────────────────────
if all_results:
    print("\n[5/5] 错误样本分析...")
    # 用性能最好的模型做错误分析
    best = max(all_results, key=lambda k: all_results[k]['binary_acc'])
    r = all_results[best]
    errors = []
    for i, (true, pred) in enumerate(zip(y_test, r['preds'])):
        if true != pred:
            errors.append({
                '文本': X_test[i],
                '真实': '正面' if true == 1 else '负面',
                '预测': '正面' if pred == 1 else '负面',
                '置信度': f"{r['scores'][i]:.1%}",
            })

    error_df = pd.DataFrame(errors)
    error_path = 'models/roberta_errors_v2.csv'
    error_df.to_csv(error_path, index=False, encoding='utf-8-sig')
    print(f"  ❌ {best} 错误: {len(errors)}/{len(y_test)} (错误率: {len(errors)/len(y_test):.2%})")
    print(f"  📄 已保存至: {error_path}")

    # 预览
    print(f"\n  --- 错误样本预览 ---")
    for _, row in error_df.head(5).iterrows():
        txt = row['文本'][:50] + ('...' if len(row['文本']) > 50 else '')
        print(f"  [{row['真实']} → {row['预测']} | {row['置信度']}] {txt}")

print("\n" + "=" * 60)
print("  ✅ 评估完成！")
print("=" * 60)