"""
中文情感分析系统 (统一版)
运行: streamlit run app.py
一次分析 = 情感预测 + LIME解释 + 谐音检测 + 讽刺识别 + 集成投票
"""
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pickle, jieba, pandas as pd, numpy as np, time, os
from datetime import datetime

# ── 页面配置 ────────────────────────────────────
st.set_page_config(page_title="中文情感分析", page_icon="🔍", layout="centered")

# ── Session State ──────────────────────────────
for k, v in [('dark_mode', False), ('history', []), ('result_cache', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

dark = st.session_state['dark_mode']
C = {'bg': '#1e1e1e' if dark else '#fff', 'tc': '#ddd' if dark else '#333',
     'card': '#2d2d2d' if dark else '#fafafa', 'border': '#444' if dark else '#e5e7eb'}

st.markdown(f"""<style>
    body{{background:{C['bg']};color:{C['tc']};}}
    .result-box{{border-radius:16px;padding:20px;text-align:center;margin:12px 0;animation:fadeIn .5s;}}
    @keyframes fadeIn{{from{{opacity:0;transform:translateY(10px);}}to{{opacity:1;transform:translateY(0);}}}}
    .highlight-text{{font-size:17px;line-height:2.2;padding:16px;background:{C['card']};border-radius:12px;border:1px solid {C['border']};}}
    .stTextArea textarea{{font-size:16px;}}
</style>""", unsafe_allow_html=True)

# ── 加载模型 ────────────────────────────────────
@st.experimental_singleton
def load_all():
    models = {}
    for path, key in [('./models/roberta-finetuned', 'roberta'),
                       ('./models/bert-finetuned', 'bert')]:
        if os.path.exists(path):
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForSequenceClassification.from_pretrained(path)
            models[key] = {'name': 'RoBERTa-wwm' if key=='roberta' else 'BERT-base',
                           'pipe': pipeline("text-classification", model=mdl, tokenizer=tok,
                                            device=-1, truncation=True, max_length=512, top_k=None)}
    if os.path.exists('models/ml_model.pkl'):
        with open('models/ml_model.pkl', 'rb') as f: clf = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f: vec = pickle.load(f)
        models['ml'] = {'name': 'LinearSVM', 'clf': clf, 'vec': vec}
    return models

models = load_all()

# ── 从深化模块导入谐音库 + 讽刺引擎（单一数据源）──
import importlib.util as _iu
_spec = _iu.spec_from_file_location("slang_mod", "07_slang_normalizer.py")
_slang_mod = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_slang_mod)

SLANG = _slang_mod.SLANG_DICT
SORTED_SLANG = sorted(SLANG.keys(), key=len, reverse=True)
_sarcasm_detector = _slang_mod.SarcasmDetector()

# ── 侧边栏 ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 设置")
    if st.checkbox("🌙 深色模式", value=dark):
        st.session_state['dark_mode'] = True
    else:
        st.session_state['dark_mode'] = False

    st.markdown("---")
    model_names = list(models.keys())
    model_choice = st.radio("推理模型", ["🤝 集成投票"] + [f"🤖 {models[k]['name']}" for k in model_names],
                            index=0, key="model")
    st.markdown("---")
    st.caption("分析模式")
    batch_mode = st.checkbox("📋 批量分析模式")

    st.markdown("---")
    for k, v in models.items(): st.success(f"✅ {v['name']}")
    if st.button("🗑 清空历史"): st.session_state['history'] = []; st.experimental_rerun()

# ── 预测函数 ────────────────────────────────────
def predict_one(text, model_key):
    info = models[model_key]
    if 'pipe' in info:
        r = info['pipe'](text)[0]
        if isinstance(r, dict): label, score = r['label'], r['score']
        else: best = max(r, key=lambda x: x['score']); label, score = best['label'], best['score']
        pred = 1 if ('LABEL_1' in str(label) or label == '1') else 0
    else:
        cut = ' '.join(jieba.cut(text)); X = info['vec'].transform([cut])
        pred = info['clf'].predict(X)[0]
        score = info['clf'].predict_proba(X)[0][pred] if hasattr(info['clf'], 'predict_proba') \
                else 1/(1+np.exp(-abs(info['clf'].decision_function(X)[0])))
    return pred, float(score)

def predict_ensemble(text):
    votes_p, votes_n, details, scores = 0, 0, [], []
    for k, v in models.items():
        p, s = predict_one(text, k)
        details.append({'model': v['name'], 'pred': p, 'score': s})
        scores.append(s if p == 1 else 1-s)
        if p == 1: votes_p += 1
        else: votes_n += 1
    return (1 if votes_p >= votes_n else 0), np.mean(scores), max(votes_p, votes_n)/len(models), details

def classify_ternary(pred, score):
    if pred == 1 and score > 0.65: return 2, "正面", "😊", "#22c55e"
    elif pred == 0 and score > 0.55: return 0, "负面", "😞", "#ef4444"   # 负面更敏感: 0.65→0.55
    else: return 1, "中性", "😐", "#f59e0b"

# ── 深度分析引擎 ──────────────────────────────
def deep_analyze(text):
    """统一分析管道：谐音→预测→讽刺→解释"""
    result = {'original': text, 'slang_hits': [], 'normalized': text}

    # 1. 谐音标准化
    for slang in SORTED_SLANG:
        if slang in result['normalized']:
            result['normalized'] = result['normalized'].replace(slang, SLANG[slang])
            result['slang_hits'].append((slang, SLANG[slang]))

    # 2. 模型预测
    clean = result['normalized']
    use_ens = model_choice.startswith("🤝")
    if use_ens:
        pred, score, agreement, details = predict_ensemble(clean)
    else:
        for k, v in models.items():
            if model_choice.endswith(v['name']):
                pred, score = predict_one(clean, k); break
        else: pred, score = 0, 0.5
        details = []
        for k, v in models.items():
            p, s = predict_one(clean, k)
            details.append({'model': v['name'], 'pred': p, 'score': s})
        votes_p = sum(1 for d in details if d['pred'] == 1)
        agreement = max(votes_p, len(details)-votes_p) / len(details)

    result['pred'] = pred
    result['score'] = score
    result['details'] = details
    result['agreement'] = agreement

    # 3. 讽刺检测（使用深化版检测器）
    pos_score = score if pred == 1 else 1 - score
    is_sar, sar_conf, sar_reasons = _sarcasm_detector.detect(text, pos_score)
    result['sarcasm'] = is_sar
    result['sarcasm_score'] = sar_conf
    result['sarcasm_reasons'] = sar_reasons

    # 4. 最终判断
    if result['sarcasm'] and result['sarcasm_score'] > 0.2:  # 更易翻转: 0.3→0.2
        result['final_pred'] = 1 - pred
        result['flipped'] = True
    else:
        result['final_pred'] = pred
        result['flipped'] = False

    tc, sent, emoji, color = classify_ternary(result['final_pred'], result['score'])
    result['ternary'] = tc
    result['sentiment'] = sent
    result['emoji'] = emoji
    result['color'] = color

    return result

# ── LIME 解释 ──────────────────────────────────
def lime_analyze(text):
    """LIME 词贡献分析"""
    from lime.lime_text import LimeTextExplainer
    jieba_tokens = [w for w in jieba.cut(text) if w.strip()]

    pipe = models.get('roberta', models.get('bert', {}))
    if 'pipe' not in pipe:
        return None, None, []

    def predictor(texts):
        results = pipe['pipe'](list(texts))
        probs = []
        for r in results:
            scores = {str(s['label']): s['score'] for s in (r if isinstance(r, list) else [r])}
            neg = scores.get('LABEL_0', scores.get('0', 0))
            pos = scores.get('LABEL_1', scores.get('1', 0))
            total = neg + pos
            probs.append([neg/total, pos/total] if total > 0 else [0.5, 0.5])
        return np.array(probs)

    explainer = LimeTextExplainer(class_names=['负面', '正面'], split_expression=lambda t: [w for w in jieba.cut(t) if w.strip()])
    exp = explainer.explain_instance(text, predictor, num_features=10, num_samples=200)

    contributions = exp.as_list()
    max_w = max(abs(w) for _, w in contributions) if contributions else 1

    token_color = {}
    for w, weight in contributions:
        intensity = min(abs(weight)/max(1, max_w), 1.0)
        token_color[w] = f'rgba(34,197,94,{intensity:.2f})' if weight > 0 else f'rgba(239,68,68,{intensity:.2f})'

    html_parts = []
    for token in jieba_tokens:
        c = token_color.get(token)
        if c:
            html_parts.append(f'<span style="background:{c};padding:3px 6px;border-radius:5px;font-weight:bold;margin:1px;">{token}</span>')
        else:
            html_parts.append(f'<span style="color:#999;">{token}</span>')
    highlighted = ''.join(html_parts)

    return exp, highlighted, contributions

# ── 结果渲染 ──────────────────────────────────
def render_result(r, elapsed):
    """渲染统一分析结果"""
    tc, sent, emoji, color = r['ternary'], r['sentiment'], r['emoji'], r['color']
    score_pct = r['score'] * 100

    # 强度等级
    if score_pct >= 80: intensity = "🔥 非常确信"
    elif score_pct >= 60: intensity = "👍 比较确信"
    else: intensity = "🤔 不太确定"

    # 主卡片
    tags = []
    if r['slang_hits']: tags.append(f"🔤 识别谐音×{len(r['slang_hits'])}")
    if r['flipped']: tags.append("🔄 讽刺翻转")
    if r.get('agreement'): tags.append(f"🤝 一致性{r['agreement']:.0%}")
    tag_html = ' · '.join(tags)

    st.markdown(f"""
    <div class="result-box" style="border:2px solid {color};">
        <div style="font-size:56px;">{emoji}</div>
        <div style="font-size:28px;font-weight:bold;color:{color};margin:8px 0;">{sentiment} · {intensity}</div>
        <div style="font-size:14px;color:#888;margin-bottom:8px;">情感强度: <b>{score_pct:.0f}/100</b></div>
        <div style="background:#e5e7eb;border-radius:10px;height:12px;max-width:300px;margin:0 auto;">
            <div style="background:{color};border-radius:10px;height:12px;width:{score_pct}%;transition:width .8s ease;"></div>
        </div>
        <div style="font-size:12px;color:#999;margin-top:8px;">{tag_html} · {elapsed:.0f}ms</div>
    </div>
    """, unsafe_allow_html=True)

    # 谐音检测
    if r['slang_hits']:
        with st.expander(f"🔤 谐音/网络用语识别 ({len(r['slang_hits'])} 处)"):
            for s, t in r['slang_hits']:
                st.markdown(f"- **{s}** → {t}")
            st.caption(f"标准化文本: _{r['normalized']}_")

    # 讽刺检测
    if r['flipped']:
        with st.expander(f"🎭 讽刺检测 (置信度 {r['sarcasm_score']:.0%})"):
            st.warning("⚠️ 检测到讽刺信号，已翻转情感判断")
            for reason in r['sarcasm_reasons']:
                st.markdown(f"- {reason}")

    # 集成投票
    with st.expander("🤝 多模型对比"):
        rows = []
        for d in r['details']:
            em = "😊" if d['pred']==1 else "😞"
            rows.append(f"| {d['model']} | {em} {'正面' if d['pred']==1 else '负面'} | {d['score']:.1%} |")
        st.markdown("| 模型 | 预测 | 置信度 |\n|------|------|--------|\n"+"\n".join(rows))
        if r.get('agreement'):
            st.caption(f"一致性: {r['agreement']:.0%}")

# ── 主界面 ────────────────────────────────────
st.markdown('<h1 style="text-align:center;">🔍 中文情感分析系统</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#888;">RoBERTa-wwm-ext · 多模型集成 · LIME 可解释 · 谐音/讽刺识别</p>', unsafe_allow_html=True)

if not batch_mode:
    # ===== 单条分析 =====
    c1, c2 = st.columns([5, 1])
    with c1:
        text = st.text_area("输入中文文本", placeholder="试试: 针不戳，这家店的服务真是太好了，等了两个小时才上菜",
                            height=120, key="main_text")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 分析")

    # 示例
    with st.expander("📌 试试这些 tricky 语句"):
        examples = [
            "针不戳，这个手机拍照绝绝子",
            "集美们避雷！这家店的服务真是太好了，等了两个小时才上菜",
            "多亏了快递，我的包裹一周后才到呢",
            "蓝瘦香菇，等了半天结果告诉我卖完了",
            "u1s1，这质量yyds，价格也良心",
            "太难吃了，从来没吃过这么差的",
        ]
        cols = st.columns(3)
        for i, ex in enumerate(examples):
            with cols[i % 3]:
                if st.button(ex[:30]+"...", key=f"ex{i}"):
                    st.session_state['main_text'] = ex
                    st.experimental_rerun()

    if analyze_btn and st.session_state.get('main_text', '').strip():
        text = st.session_state['main_text'].strip()
        with st.spinner("🧠 深度分析中..."):
            t0 = time.time()
            r = deep_analyze(text)
            elapsed = (time.time() - t0) * 1000

        render_result(r, elapsed)

        # LIME 解释
        try:
            exp, highlighted, contribs = lime_analyze(text)
            if highlighted:
                st.subheader("🔑 LIME 词贡献分析")
                chart_data = [{'关键词': w, '贡献': c} for w, c in contribs]
                st.bar_chart(pd.DataFrame(chart_data).set_index('关键词'))
                st.markdown(f'<div class="highlight-text">{highlighted}</div>', unsafe_allow_html=True)
        except Exception:
            pass  # LIME 失败时静默

        # 入历史
        _, sent, _, _ = classify_ternary(r['final_pred'], r['score'])
        st.session_state['history'].insert(0, {
            'time': datetime.now().strftime('%H:%M:%S'), 'text': text[:80],
            'sentiment': sent, 'score': r['score'],
        })
        if len(st.session_state['history']) > 20:
            st.session_state['history'] = st.session_state['history'][:20]

else:
    # ===== 批量分析 =====
    with st.form("batch"):
        batch_text = st.text_area("每行一条文本", placeholder="这家店很好\n服务太差了\n还行吧",
                                  height=200, key="btext")
        submitted = st.form_submit_button("🔍 批量分析")

    if submitted and st.session_state.get('btext', '').strip():
        texts = [t.strip() for t in st.session_state['btext'].strip().split('\n') if t.strip()]
        if texts:
            results, pos_c, neu_c, neg_c, scores_list = [], 0, 0, 0, []
            progress_bar, status = st.progress(0), st.empty()

            for i, t in enumerate(texts):
                status.text(f"分析中... {i+1}/{len(texts)}")
                r = deep_analyze(t)
                tc, sent, emoji, _ = r['ternary'], r['sentiment'], r['emoji'], r['color']
                if tc == 2: pos_c += 1
                elif tc == 1: neu_c += 1
                else: neg_c += 1
                scores_list.append(r['score'] if r['final_pred']==1 else 1-r['score'])
                results.append({"序号": i+1, "文本": t[:50]+('...' if len(t)>50 else ''),
                                "情感": f"{emoji} {sent}", "强度": f"{r['score']:.0%}",
                                "谐音": f"{len(r['slang_hits'])}处" if r['slang_hits'] else "",
                                "讽刺": "🎭" if r['flipped'] else ""})
                progress_bar.progress((i+1)/len(texts))

            status.text("✅ 完成!"); progress_bar.empty()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("正面 😊", pos_c); c2.metric("中性 😐", neu_c)
            c3.metric("负面 😞", neg_c); c4.metric("讽刺/谐音", sum(1 for r in results if r['讽刺'] or r['谐音']))

            st.bar_chart(pd.DataFrame({'情感': ['正面 😊','中性 😐','负面 😞'], '数量': [pos_c, neu_c, neg_c]}).set_index('情感'))

            if len(scores_list) > 1:
                st.subheader("📈 情感趋势")
                st.line_chart(pd.DataFrame({'序号': range(1, len(scores_list)+1), '强度': [s*100 for s in scores_list]}).set_index('序号'))

            df = pd.DataFrame(results); st.dataframe(df)
            st.download_button("📥 下载 CSV", df.to_csv(index=False, encoding='utf-8-sig'), "result.csv", key="dl")

# ── 历史 ────────────────────────────────────
if st.session_state['history']:
    with st.expander(f"📋 分析历史 ({len(st.session_state['history'])} 条)"):
        for item in st.session_state['history'][:8]:
            em_map = {'正面': '😊', '负面': '😞', '中性': '😐'}
            st.markdown(f"""<div style="padding:6px 10px;border-left:3px solid{'#22c55e' if item['sentiment']=='正面'else'#ef4444' if item['sentiment']=='负面'else'#f59e0b'};margin:4px 0;font-size:14px;background:{C['card']};border-radius:0 8px 8px 0;">
                {em_map.get(item['sentiment'],'')} <b>{item['sentiment']}</b> ({item['score']:.1%}) · <span style="color:#888;">{item['time']}</span>
                <br><span style="color:#888;">{item['text']}</span></div>""", unsafe_allow_html=True)

st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.caption("🧠 RoBERTa-wwm-ext + BERT + LinearSVM")
c2.caption("🔬 LIME 可解释 · 🎭 讽刺检测 · 🔤 谐音识别")
c3.caption("⚡ Streamlit")
