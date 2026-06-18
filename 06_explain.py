"""
步骤6：模型可解释性分析
- LIME 解释器：高亮对预测贡献最大的词
- 关键词提取：TextRank + TF-IDF
- 方面-情感对：抽取关键短语及其情感倾向
"""
# ⚠️ 必须在导入前设置镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import jieba
import jieba.analyse
import pickle
import re
from collections import Counter

# ── 模型加载 ────────────────────────────────────
def load_all_models():
    """加载所有可用模型"""
    models = {}

    # ML
    try:
        with open('models/ml_model.pkl', 'rb') as f:
            clf = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vec = pickle.load(f)
        models['ml'] = {'type': 'ml', 'clf': clf, 'vec': vec}
    except FileNotFoundError:
        pass

    # BERT
    try:
        bp = 'models/bert-finetuned'
        if os.path.exists(bp):
            tok = AutoTokenizer.from_pretrained(bp)
            mdl = AutoModelForSequenceClassification.from_pretrained(bp)
            models['bert'] = {'type': 'nn', 'pipe': pipeline(
                "text-classification", model=mdl, tokenizer=tok,
                device=-1, truncation=True, max_length=512, return_all_scores=True)}
    except Exception:
        pass

    # RoBERTa
    try:
        rp = 'models/roberta-finetuned'
        if os.path.exists(rp):
            tok = AutoTokenizer.from_pretrained(rp)
            mdl = AutoModelForSequenceClassification.from_pretrained(rp)
            models['roberta'] = {'type': 'nn', 'pipe': pipeline(
                "text-classification", model=mdl, tokenizer=tok,
                device=-1, truncation=True, max_length=512, return_all_scores=True)}
    except Exception:
        pass

    return models

# ── 1. LIME 可解释性 ───────────────────────────
def explain_with_lime(text, model_type='roberta', num_features=10):
    """用 LIME 解释模型预测，返回 (词贡献列表, 可视化HTML)"""
    from lime.lime_text import LimeTextExplainer

    models = load_all_models()
    if model_type not in models:
        return [], "<p>模型未加载</p>"

    info = models[model_type]

    # 定义预测函数给 LIME
    if info['type'] == 'nn':
        class_names = ['负面', '正面']
        pipe = info['pipe']

        def predictor(texts):
            results = pipe(list(texts))
            probs = []
            for r in results:
                # return_all_scores 返回 [{'label': ..., 'score': ...}, ...]
                scores = {s['label']: s['score'] for s in r}
                neg = scores.get('LABEL_0', scores.get('0', 0))
                pos = scores.get('LABEL_1', scores.get('1', 0))
                # 归一化
                total = neg + pos
                if total > 0:
                    probs.append([neg/total, pos/total])
                else:
                    probs.append([0.5, 0.5])
            return np.array(probs)

    else:
        class_names = ['负面', '正面']
        vec = info['vec']
        clf = info['clf']

        def predictor(texts):
            cut = [' '.join(jieba.cut(t)) for t in texts]
            X = vec.transform(cut)
            if hasattr(clf, 'predict_proba'):
                return clf.predict_proba(X)
            else:
                d = clf.decision_function(X)
                p = np.zeros((len(d), 2))
                p[:, 1] = 1.0 / (1.0 + np.exp(-d))
                p[:, 0] = 1.0 - p[:, 1]
                return p

    # 创建 LIME 解释器
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predictor, num_features=num_features,
                                      num_samples=200)

    # 提取词贡献
    word_contributions = []
    for word, weight in exp.as_list():
        word_contributions.append({
            'word': word,
            'weight': weight,
            'sentiment': 'positive' if weight > 0 else 'negative',
        })

    # 生成高亮 HTML
    highlighted_html = _highlight_text(text, word_contributions)
    label_idx = exp.predict_proba.argmax()
    confidence = exp.predict_proba[label_idx]

    return word_contributions, highlighted_html, class_names[label_idx], confidence

def _highlight_text(text, word_contributions):
    """根据词贡献生成高亮 HTML"""
    # 构建词→颜色映射
    word_colors = {}
    max_weight = max(abs(w['weight']) for w in word_contributions) if word_contributions else 1

    for w in word_contributions:
        intensity = min(abs(w['weight']) / max_weight, 1.0)
        if w['weight'] > 0:
            # 正面→绿色
            alpha = int(60 + 195 * intensity)
            word_colors[w['word']] = f'rgba(34,197,94,{intensity:.1f})'
        else:
            # 负面→红色
            alpha = int(60 + 195 * intensity)
            word_colors[w['word']] = f'rgba(239,68,68,{intensity:.1f})'

    # 分词后用颜色包裹
    words = list(jieba.cut(text))
    html_parts = []
    for word in words:
        if word.strip():
            color = word_colors.get(word)
            if color:
                html_parts.append(
                    f'<span style="background:{color}; padding:2px 4px; '
                    f'border-radius:4px; font-weight:bold;" '
                    f'title="贡献度: {word}">{word}</span>'
                )
            else:
                html_parts.append(f'<span style="color:#999;">{word}</span>')
    return ' '.join(html_parts)

# ── 2. 关键词 & 方面提取 ───────────────────────
def extract_keywords(text, topk=8):
    """用 TextRank + TF-IDF 提取关键词"""
    # TextRank
    tr_keywords = jieba.analyse.textrank(text, topK=topk, withWeight=True)
    # TF-IDF
    tfidf_keywords = jieba.analyse.extract_tags(text, topK=topk, withWeight=True)

    # 合并去重
    keywords = {}
    for w, score in tr_keywords:
        keywords[w] = keywords.get(w, 0) + score * 0.5
    for w, score in tfidf_keywords:
        keywords[w] = keywords.get(w, 0) + score * 0.5

    sorted_kw = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    return [{'word': w, 'score': round(s, 4)} for w, s in sorted_kw[:topk]]

def extract_aspects(text, models, topk=5):
    """提取方面词（名词/动词短语）并标注情感"""
    # 用 jieba 词性标注提取名词和动词
    import jieba.posseg as pseg
    words = pseg.cut(text)

    aspects = []
    for w, flag in words:
        # 名词、动词、形容词作为候选方面
        if flag.startswith(('n', 'v', 'a')) and len(w) >= 2:
            aspects.append(w)

    # 去重
    aspects = list(dict.fromkeys(aspects))[:topk]

    # 对每个方面判断情感
    if 'roberta' in models:
        pipe = models['roberta']['pipe']
    elif 'bert' in models:
        pipe = models['bert']['pipe']
    else:
        pipe = None

    results = []
    for aspect in aspects:
        if pipe:
            r = pipe(aspect)[0]
            # 找最高分的标签
            best = max(r, key=lambda x: x['score'])
            label = best['label']
            if 'LABEL_1' in str(label) or label == '1':
                sent = 'positive'
            else:
                sent = 'negative'
            conf = best['score']
        else:
            sent, conf = 'neutral', 0.5

        results.append({'aspect': aspect, 'sentiment': sent, 'confidence': round(conf, 3)})

    return results

# ── 3. 模型集成投票 ────────────────────────────
def ensemble_predict(text, models):
    """多模型集成投票"""
    votes = []
    details = {}

    for name, info in models.items():
        if info['type'] == 'nn':
            r = info['pipe'](text)[0]
            best = max(r, key=lambda x: x['score'])
            label = best['label']
            score = best['score']
            if 'LABEL_1' in str(label) or label == '1':
                pred = 1
            else:
                pred = 0
        else:
            cut = ' '.join(jieba.cut(text))
            X = info['vec'].transform([cut])
            pred = info['clf'].predict(X)[0]
            if hasattr(info['clf'], 'predict_proba'):
                score = info['clf'].predict_proba(X)[0][pred]
            else:
                d = abs(info['clf'].decision_function(X)[0])
                score = 1.0 / (1.0 + np.exp(-d))

        votes.append(pred)
        details[name] = {'pred': pred, 'score': round(float(score), 4)}

    # 多数投票
    pos_votes = sum(votes)
    neg_votes = len(votes) - pos_votes
    final_pred = 1 if pos_votes > neg_votes else 0
    agreement = max(pos_votes, neg_votes) / len(votes)

    return final_pred, agreement, details

# ── 4. 情感强度量化 ────────────────────────────
def sentiment_intensity(text, models):
    """量化情感强度 0-100（100=极正面, 0=极负面, 50=中性）"""
    pred, agreement, details = ensemble_predict(text, models)

    # 收集各模型分数
    scores = []
    for name, info in models.items():
        if info['type'] == 'nn':
            r = info['pipe'](text)[0]
            for s in r:
                if 'LABEL_1' in str(s['label']) or s['label'] == '1':
                    scores.append(s['score'])
        else:
            cut = ' '.join(jieba.cut(text))
            X = info['vec'].transform([cut])
            if hasattr(info['clf'], 'predict_proba'):
                scores.append(info['clf'].predict_proba(X)[0][1])
            else:
                d = info['clf'].decision_function(X)[0]
                scores.append(1.0 / (1.0 + np.exp(-d)))

    avg_score = np.mean(scores) if scores else 0.5
    intensity = round(avg_score * 100)

    if intensity >= 70:
        level = "强烈正面"
    elif intensity >= 55:
        level = "偏正面"
    elif intensity >= 45:
        level = "中性"
    elif intensity >= 30:
        level = "偏负面"
    else:
        level = "强烈负面"

    return intensity, level, agreement

# ── CLI 测试 ────────────────────────────────────
if __name__ == '__main__':
    text = "这家店的服务态度非常好，菜品也很美味，就是价格有点贵"
    print("=" * 60)
    print("         可解释性分析测试")
    print("=" * 60)
    print(f"\n输入: {text}\n")

    models = load_all_models()
    print(f"可用模型: {list(models.keys())}")

    if models:
        # LIME
        print("\n--- LIME 词贡献 ---")
        contributions, html, label, conf = explain_with_lime(text, list(models.keys())[-1])
        for w in contributions:
            bar = '█' * int(abs(w['weight']) * 30)
            print(f"  {'+' if w['weight']>0 else '-'} {w['word']:<10} {bar} ({w['weight']:.3f})")

        # 关键词
        print("\n--- 关键词 ---")
        for kw in extract_keywords(text):
            print(f"  {kw['word']}: {kw['score']:.4f}")

        # 集成
        print("\n--- 集成投票 ---")
        pred, agreement, details = ensemble_predict(text, models)
        final = "正面" if pred == 1 else "负面"
        print(f"  结果: {final} (一致性: {agreement:.0%})")
        for name, d in details.items():
            label = "正面" if d['pred'] == 1 else "负面"
            print(f"    {name}: {label} ({d['score']:.1%})")

        # 强度
        intensity, level, _ = sentiment_intensity(text, models)
        print(f"\n--- 情感强度 ---")
        print(f"  分数: {intensity}/100 ({level})")
