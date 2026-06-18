"""
步骤2：传统机器学习基线
- TF-IDF 特征提取
- 逻辑回归 / SVM / 朴素贝叶斯 对比
- 保存最优模型
"""
# ⚠️ 必须在导入 datasets 前设置镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
import pickle
import jieba
import time

os.makedirs('models', exist_ok=True)

# ── 1. 加载数据 ──────────────────────────────────
print("正在加载数据集...")
dataset = load_dataset("lansinuote/ChnSentiCorp")

# 合并训练集和验证集作为训练数据，测试集留作评估
X_train_raw = list(dataset['train']['text']) + list(dataset['validation']['text'])
y_train = list(dataset['train']['label']) + list(dataset['validation']['label'])
X_test_raw = list(dataset['test']['text'])
y_test = list(dataset['test']['label'])

print(f"训练集: {len(X_train_raw)} 条")
print(f"测试集: {len(X_test_raw)} 条")

# ── 2. 中文分词 ──────────────────────────────────
print("\n正在进行中文分词（jieba）...")
t0 = time.time()

def tokenize(text):
    return ' '.join(jieba.cut(text))

X_train = [tokenize(t) for t in X_train_raw]
X_test = [tokenize(t) for t in X_test_raw]

print(f"分词完成，耗时: {time.time() - t0:.1f}s")

# ── 3. 三个模型对比 ─────────────────────────────
models = {
    'LogisticRegression': LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
    ),
    'LinearSVM': LinearSVC(
        max_iter=2000,
        C=1.0,
        class_weight='balanced',
    ),
    'NaiveBayes': MultinomialNB(alpha=0.5),
}

best_score = 0
best_name = None
best_pipe = None
all_results = {}

print("\n" + "=" * 60)
print("                  开始训练 & 评估")
print("=" * 60)

for name, model in models.items():
    print(f"\n>>> 训练 {name} ...")
    t0 = time.time()

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),       # unigram + bigram
            max_features=15000,       # 保留最多特征词数
            min_df=2,                  # 至少出现 2 次
        )),
        ('clf', model),
    ])

    pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    # 评估
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"  训练耗时: {train_time:.1f}s")
    print(f"  测试准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  测试 Macro F1: {f1:.4f}")
    print(f"\n  分类报告:")
    print(classification_report(y_test, y_pred, target_names=['负面', '正面']))

    all_results[name] = {'acc': acc, 'f1': f1}

    if acc > best_score:
        best_score = acc
        best_name = name
        best_pipe = pipe

# ── 4. 保存最优模型 ─────────────────────────────
print("\n" + "=" * 60)
print("                  结果汇总")
print("=" * 60)

for name, metrics in all_results.items():
    marker = " ⭐ 最优" if name == best_name else ""
    print(f"  {name}: 准确率={metrics['acc']:.4f}, F1={metrics['f1']:.4f}{marker}")

# 保存
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(best_pipe.named_steps['tfidf'], f)
with open('models/ml_model.pkl', 'wb') as f:
    pickle.dump(best_pipe.named_steps['clf'], f)

print(f"\n✅ 最优模型: {best_name} (准确率: {best_score:.4f})")
print(f"✅ 已保存至 models/tfidf_vectorizer.pkl 和 models/ml_model.pkl")
