"""
步骤1：数据探索
- 加载 chnsenticorp 数据集
- 查看样本
- 统计标签分布
- 检查文本长度分布
"""
# ⚠️ 必须在导入 datasets 前设置镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体（防止图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('data', exist_ok=True)

print("正在加载 ChnSentiCorp 数据集...")
dataset = load_dataset("lansinuote/ChnSentiCorp")

print("\n" + "=" * 60)
print("                     数据集结构")
print("=" * 60)
print(dataset)
print(f"\n各 split 样本数:")
for split in dataset:
    print(f"  {split}: {len(dataset[split])} 条")

# ── 查看样本 ────────────────────────────────────
for split in dataset:
    print(f"\n" + "-" * 40)
    print(f"  [{split} 集] 前 5 条样本")
    print("-" * 40)
    df = dataset[split].to_pandas()
    for i, row in df.head(5).iterrows():
        label_text = "正面 👍" if row['label'] == 1 else "负面 👎"
        txt = row['text'][:60] + ('...' if len(row['text']) > 60 else '')
        print(f"  [{label_text}] {txt}")

# ── 标签分布 ────────────────────────────────────
print("\n" + "=" * 60)
print("                     标签分布")
print("=" * 60)

df_train = dataset['train'].to_pandas()
df_val = dataset['validation'].to_pandas()
df_test = dataset['test'].to_pandas()

print(f"\n训练集 ({len(df_train)} 条):")
label_counts = df_train['label'].value_counts()
for label, count in label_counts.items():
    label_name = "正面(1)" if label == 1 else "负面(0)"
    pct = count / len(df_train) * 100
    print(f"  {label_name}: {count} 条 ({pct:.1f}%)")

print(f"\n验证集 ({len(df_val)} 条):")
label_counts_val = df_val['label'].value_counts()
for label, count in label_counts_val.items():
    label_name = "正面(1)" if label == 1 else "负面(0)"
    pct = count / len(df_val) * 100
    print(f"  {label_name}: {count} 条 ({pct:.1f}%)")

print(f"\n测试集 ({len(df_test)} 条):")
label_counts_test = df_test['label'].value_counts()
for label, count in label_counts_test.items():
    label_name = "正面(1)" if label == 1 else "负面(0)"
    pct = count / len(df_test) * 100
    print(f"  {label_name}: {count} 条 ({pct:.1f}%)")

# ── 文本长度统计 ────────────────────────────────
df_train['text_len'] = df_train['text'].apply(len)
print(f"\n文本长度统计 (训练集):")
print(df_train['text_len'].describe())
print(f"  - 最短: {df_train['text_len'].min()} 字")
print(f"  - 最长: {df_train['text_len'].max()} 字")
print(f"  - 中位数: {df_train['text_len'].median():.0f} 字")
print(f"  - 75% 分位数: {df_train['text_len'].quantile(0.75):.0f} 字")
print(f"  - 95% 分位数: {df_train['text_len'].quantile(0.95):.0f} 字")

# ── 画图 ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图1：标签分布
colors = ['#ef4444', '#22c55e']
axes[0].bar(['负面 (0)', '正面 (1)'], label_counts.values, color=colors, edgecolor='white')
axes[0].set_title('训练集标签分布', fontsize=14, fontweight='bold')
axes[0].set_ylabel('样本数')
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# 子图2：文本长度分布
axes[1].hist(df_train['text_len'], bins=50, color='#3b82f6', edgecolor='white', alpha=0.8)
axes[1].axvline(df_train['text_len'].median(), color='red', linestyle='--', label=f"中位数: {df_train['text_len'].median():.0f}")
axes[1].axvline(128, color='orange', linestyle='--', label='BERT max_length=128')
axes[1].set_title('文本长度分布', fontsize=14, fontweight='bold')
axes[1].set_xlabel('字符数')
axes[1].set_ylabel('样本数')
axes[1].legend()

plt.tight_layout()
save_path = 'data/data_explore.png'
plt.savefig(save_path, dpi=120)
print(f"\n📊 图表已保存至: {save_path}")
plt.close()

print("\n" + "=" * 60)
print("  ✅ 数据探索完成！")
print("=" * 60)
