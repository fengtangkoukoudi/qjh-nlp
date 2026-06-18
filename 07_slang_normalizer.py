"""
中文谐音/网络用语标准化 + 讽刺检测（深化版）
- 200+ 词条覆盖 2024-2026 主流网络表达
- 多维度讽刺检测：双句对照 · emoji 信号 · 领域特化 · 语境评分
"""
import re
import jieba

# ============================================================
#  第一部分：谐音/网络用语标准化词典 (200+ 词条)
# ============================================================
SLANG_DICT = {
    # ── 拼音谐音 (2024-2026) ──
    "针不戳": "真不错",     "集美": "姐妹",         "蚌埠住了": "绷不住了",
    "绷不住": "忍不住",     "蓝瘦": "难受",         "香菇": "想哭",
    "神马": "什么",         "有木有": "有没有",     "酱紫": "这样子",
    "表": "不要",           "造": "知道",           "宣": "喜欢",
    "鸡冻": "激动",         "捉急": "着急",         "涨姿势": "长知识",
    "油菜花": "有才华",     "童鞋": "同学",         "盆友": "朋友",
    "妹纸": "妹子",         "汉纸": "汉子",         "骚年": "少年",
    "小公举": "小公主",     "灰常": "非常",         "稀饭": "喜欢",
    "美式": "没事",         "奶思": "不错",         "古德": "好的",
    "恰饭": "赚钱",         "来迟": "来晚",         "好康": "好看",
    "母鸡": "不知道",       "猴赛雷": "好厉害",

    # ── 新兴网络热词 (2024-2026) ──
    "绝绝子": "非常好",     "无语子": "无语",       "好家伙": "真厉害",
    "遥遥领先": "非常先进", "遥遥落后": "非常落后", "遥遥无期": "没有希望",
    "显眼包": "引人注目",   "特种兵": "高强度",     "多巴胺": "快乐",
    "搭子": "伙伴",         "city不city": "时尚吗", "尊嘟假嘟": "真的假的",
    "i人": "内向的人",     "e人": "外向的人",     "淡人": "佛系的人",
    "纯纯": "完全",         "纯纯无语": "完全无语", "纯纯大冤种": "完全被骗",
    "家人们谁懂啊": "大家理解一下", "咱就是说": "我说",

    # ── 贴吧/抽象话 ──
    "典": "太典型了",       "孝": "盲目维护",       "急": "着急了",
    "乐": "太可笑了",       "赢": "自欺欺人",       "绷": "绷不住了",
    "难绷": "难以绷住",     "哈人": "吓人",         "麻了": "无奈了",
    "抽象": "离谱",         "太抽象了": "太离谱了", "逆天": "离谱",
    "难蚌": "难以绷住",     "流汗": "无语",         "黄豆": "无语",
    "玩原神玩的": "不可理喻",

    # ── 互联网缩写/拼音缩写 ──
    "yyds": "永远的神",     "xswl": "笑死我了",     "awsl": "太可爱了",
    "u1s1": "有一说一",     "yysy": "有一说一",     "srds": "虽然但是",
    "nsdd": "你说得对",     "zqsg": "真情实感",     "dbq": "对不起",
    "pyq": "朋友圈",        "bdjw": "不懂就问",     "tql": "太强了",
    "xdm": "兄弟们",        "jms": "姐妹们",        "pljj": "漂亮姐姐",
    "sqgg": "帅气哥哥",     "hxd": "好兄弟",        "wlsw": "无聊死我了",
    "bkpp": "不可怕怕",     "yysyqs": "有一说一确实",

    # ── 情绪表达 ──
    "破防": "情绪崩溃",     "下头": "扫兴",         "上头": "沉迷兴奋",
    "emo": "情绪低落",      "破大防": "彻底崩溃",   "心态炸了": "心态崩了",
    "难顶": "难以承受",     "窒息": "令人窒息",     "裂开": "崩溃",
    "社死": "社交尴尬",     "社恐": "害怕社交",     "牛马": "苦命打工人",

    # ── 生活态度 ──
    "躺平": "放弃努力",     "摆烂": "破罐破摔",     "佛系": "随缘随意",
    "内卷": "过度竞争",     "润": "离开",           "韭": "被收割",
    "割韭菜": "被收割",     "噶韭菜": "被收割",     "大冤种": "被坑的人",
    "真香": "后悔但接受",   "打脸": "自相矛盾",     "翻车": "失败",
    "塌房": "偶像人设崩塌", "避雷": "警告避开",     "踩雷": "遇到问题",
    "种草": "推荐购买",     "拔草": "放弃购买",

    # ── 英文谐音梗 ──
    "栓Q": "感谢但无奈",    "duck不必": "大可不必", "book思议": "不可思议",
    "深藏blue": "深藏不露", "无可phone告": "无可奉告", "半tour废": "半途而废",
    "E路向北": "一路向北",  "无fuck说": "无话可说", "生无clean": "生无可恋",
    "tony带水": "拖泥带水", "贪生pass": "贪生怕死", "cheer不舍": "锲而不舍",
    "star皆空": "四大皆空", "battle不休": "战斗不休",

    # ── 数字谐音 ──
    "666": "很厉害",        "233": "哈哈哈",        "555": "呜呜呜",
    "520": "我爱你",        "886": "再见",          "9494": "是的是的",
    "1314": "一生一世",     "995": "救救我",        "748": "去死吧",
    "7456": "气死我了",     "5201314": "我爱你一生一世",

    # ── 平台黑话 ──
    "这波": "这次",         "属实": "确实",         "有一说一": "客观来说",
    "不吹不黑": "客观评价", "懂的都懂": "无需多言", "太真实了": "非常真实",
    "我麻了": "我无奈了",   "难搞": "难处理",       "离谱": "难以置信",
    "离大谱": "极其离谱",   "给力": "很棒",         "坑爹": "坑人",
    "然并卵": "没用",       "老铁": "兄弟",         "扎心": "伤心",
    "键盘侠": "网络喷子",   "杠精": "爱抬杠的人",   "柠檬精": "嫉妒的人",
    "舔狗": "讨好别人的人", "海王": "花心的人",     "社畜": "苦命上班族",

    # ── 经济/消费 ──
    "平替": "平价替代品",   "智商税": "不值得买",   "刺客": "价格陷阱",
    "雪糕刺客": "价格陷阱", "价格刺客": "价格陷阱", "消费降级": "减少消费",
    "消费升级": "提升消费",

    # ── 动作/状态 ──
    "摸鱼": "偷懒",         "划水": "偷懒",         "摆": "放弃",
    "开摆": "开始摆烂",     "冲": "努力",           "肝": "拼命做",
    "氪金": "花钱充值",     "白嫖": "免费获取",     "薅羊毛": "占便宜",
    "双标": "双重标准",     "画饼": "空口承诺",     "背锅": "承担责任",
    "甩锅": "推卸责任",     "带节奏": "引导舆论",   "控评": "控制评论",

    # ── 程度副词 ──
    "贼": "非常",           "巨": "非常",           "超": "非常",
    "爆": "极其",           "狂": "拼命地",         "疯": "疯狂地",
    "死": "极其",           "绝": "极其",
}

# ============================================================
#  第二部分：讽刺检测引擎（多维度）
# ============================================================

class SarcasmDetector:
    """多维度讽刺检测引擎"""

    def __init__(self):
        # ── A. 单句模式匹配 ──
        self.patterns = [
            # 夸张赞美 + 负面语境
            (r'(真是|实在|简直|这也)(太|很|非常|无比|绝了|过于|极其)(好|棒|厉害|牛|赞|优秀|完美|给力|到位)',
             '夸张赞美+负面语境', -0.3),
            # 感谢 + 坏结果
            (r'(多亏|幸好|感谢|托.{0,2}福|谢谢|感恩).{0,15}(?:才|最后|结果|竟然|居然|总算).+',
             '感谢模式+延迟/坏结果', -0.25),
            # 谢谢 + 损坏/坏结果
            (r'(谢谢|感谢|感恩).{0,15}(?:坏|烂|碎|摔|砸|丢|没|废|破|裂)',
             '谢谢+损坏/坏结果', -0.3),
            # 号称 vs 实际
            (r'(?:号称|说是|说什么|美其名曰|吹得).{0,10}(?:实际|结果|其实|根本|完全|压根).+',
             '宣传vs实际反差', -0.2),
            # 反问讽刺
            (r'(?:这|那)(?:也|还|就)(?:叫|算|是)(?:好|优秀|不错|可以|行|服务|态度|质量)',
             '反问式讽刺', -0.25),
            # 避雷/劝退
            (r'(?:避雷|千万别|不要买|别去|慎入|踩雷|翻车|塌房|劝退|退退退)',
             '避雷/劝退警告', -0.3),
            # 反语
            (r'(?:真|可|太|好)(?:有意思|有趣|好笑|搞笑|幽默|逗)[了哦啊]?[！!。？]*',
             '反语"有意思/有趣"', -0.1),
            # 夸张等待
            (r'(等了?|花了?|用了?|排了?)(\d+)(?:个|多)?(?:小时|天|分钟|周|月).{0,8}(?:才|就|竟然|只)',
             '夸张等待+不满结果', -0.15),
            # 价格讽刺
            (r'(?:就|才|只)(?:\d+)(?:块|元|毛|角).{0,10}(?:这|这种|这个)(?:质量|服务|态度|东西|玩意)',
             '低价低质讽刺', -0.15),
            # 极端赞美 + 但是
            (r'(?:完美|一流|绝了|无敌|太棒了|非常好|超级好).{0,15}(?:但是|不过|就是|可惜|然而)',
             '夸赞+"但是"转折', -0.2),
            # 贴吧五字真言(单字出现时结合上下文)
            (r'^[典孝急乐赢绷]$', '贴吧五字真言', -0.1),
            # XX得YY（夸张负面）
            (r'(?:好|真|太|非常|特别|超级|简直).{1,3}(?:得|的)(?:要死|要命|不行|过分|离谱|可怕|吓人|逆天)',
             '夸张负面表达', -0.15),
            # 所谓/你们说的 X？ → 质疑式讽刺
            (r'(?:这就|这就这|这算|什么叫|啥叫|所谓|你们说的).{0,10}(?:服务|态度|质量|水平|能力)[？?!！]*',
             '质疑式讽刺反问', -0.2),
        ]

        # ── B. 双句对照模式 ──
        self.two_sentence_patterns = [
            (r'(?:本来|原本|一开始|刚).+[。，；].+(?:结果|没想到|谁知道|哪知道|最后).+',
             '时间线反差', -0.2),
            (r'(?:网上|评论区|大家).+(?:都说|都说好|吹爆).+[。，；].+(?:结果|实际|其实).+',
             '口碑vs实际反差', -0.2),
        ]

        # ── C. Emoji 信号 ──
        self.sarcasm_emojis = {'🙃', '😅', '🤡', '💀', '🤣', '😬', '🥲', '🤪', '😏', '🫠'}
        self.anger_emojis = {'😡', '🤬', '💢', '😤', '👎', '🖕'}

        # ── D. 语境词库 ──
        self.negative_context = {
            '等了', '坏了', '烂了', '破了', '漏了', '掉了', '碎了',
            '投诉', '退款', '差评', '无语', '失望', '生气', '愤怒',
            '骗', '坑', '假', '垃圾', '恶心', '讨厌', '烦',
            '怎么可能', '搞笑呢', '逗我', '开玩笑', '忽悠',
            '迟到', '延误', '缺货', '瑕疵', '破损', '异味',
        }
        self.positive_extreme = {
            '非常好', '太棒了', '完美', '一流', '绝了', '无敌',
            '太好了', '超级好', '棒极了', '满分', '神了',
        }
        self.negative_words = {
            '差', '烂', '坏', '糟', '坑', '骗', '垃圾', '劣质',
            '差劲', '低劣', '恶劣', '不堪',
        }

    def detect(self, text, sentiment_score):
        """
        返回: (is_sarcasm, confidence, reasons)
        sentiment_score: 模型输出的正面置信度 (0-1)
        """
        sarcasm_score = 0.0
        reasons = []

        # ── 1. 单句模式匹配 ──
        for pattern, name, weight in self.patterns:
            if re.search(pattern, text):
                sarcasm_score += weight
                reasons.append(f"匹配模式「{name}」")

        # ── 2. 双句对照 ──
        for pattern, name, weight in self.two_sentence_patterns:
            if re.search(pattern, text):
                sarcasm_score += weight
                reasons.append(f"双句对照「{name}」")

        # ── 3. Emoji 信号 ──
        emojis_found = set()
        for ch in text:
            if ch in self.sarcasm_emojis:
                emojis_found.add(ch)
            if ch in self.anger_emojis:
                emojis_found.add(ch)
                sarcasm_score -= 0.1
                reasons.append(f"愤怒emoji {' '.join(emojis_found)}")

        # ── 4. 情感矛盾检测 ──
        words = set(jieba.cut(text))
        neg_hits = words & self.negative_context
        if neg_hits and sentiment_score > 0.65:
            sarcasm_score -= 0.25
            reasons.append(f"负面信号词 {neg_hits} + 高正面置信度({sentiment_score:.0%})矛盾")

        # ── 5. 极端正面词 + 负面词共现 ──
        pos_extreme_hits = words & self.positive_extreme
        neg_hits_all = words & self.negative_words
        if pos_extreme_hits and neg_hits_all:
            sarcasm_score -= 0.2
            reasons.append(f"极端正面词{pos_extreme_hits}+负面词{neg_hits_all}共现")

        # ── 6. 感叹号 + 负面信号 ──
        if ('！' in text or '!' in text) and neg_hits:
            sarcasm_score -= 0.05

        # ── 7. 连续问号 ──
        if '？？' in text or '??' in text:
            if sentiment_score > 0.7:
                sarcasm_score -= 0.1
                reasons.append("连续问号+高正面置信度，疑似质疑式讽刺")

        # ── 8. 省略号 + 负面词 (意味深长) ──
        if ('…' in text or '...' in text) and neg_hits_all:
            sarcasm_score -= 0.05

        is_sarcasm = sarcasm_score < -0.15
        confidence = min(abs(sarcasm_score) * 2, 0.95)
        return is_sarcasm, confidence, reasons


# ============================================================
#  第三部分：标准化 & 综合理解
# ============================================================

class SlangNormalizer:
    """谐音/网络用语 → 标准中文"""

    def __init__(self):
        self.dict = SLANG_DICT
        self.sorted_keys = sorted(self.dict.keys(), key=len, reverse=True)

    def normalize(self, text):
        """标准化并标注（用于展示），避免嵌套替换"""
        result = text
        hits = []
        replaced = set()  # 已替换的 (start, end) 区间
        for slang in self.sorted_keys:
            idx = result.find(slang)
            while idx != -1:
                end = idx + len(slang)
                overlap = any(not (end <= s or idx >= e) for s, e in replaced)
                if not overlap:
                    replaced.add((idx, end))
                    new = self.dict[slang]
                    result = result[:idx] + f"【{new}】" + result[end:]
                    hits.append((slang, new))
                idx = result.find(slang, idx + 1)
        return result, hits

    def normalize_clean(self, text):
        """纯净标准化（用于模型推理），避免嵌套替换"""
        result = text
        replaced = set()
        for slang in self.sorted_keys:
            idx = result.find(slang)
            while idx != -1:
                end = idx + len(slang)
                overlap = any(not (end <= s or idx >= e) for s, e in replaced)
                if not overlap:
                    replaced.add((idx, end))
                    new = self.dict[slang]
                    result = result[:idx] + new + result[end:]
                idx = result.find(slang, idx + 1)
        return result


class DeepUnderstander:
    """统一深度理解管道：标准化 → 预测 → 讽刺检测 → 综合判断"""

    def __init__(self, models):
        self.normalizer = SlangNormalizer()
        self.detector = SarcasmDetector()
        self.models = models

    def analyze(self, text):
        result = {
            'original': text,
            'normalized': text,
            'slang_hits': [],
            'sentiment': None, 'confidence': 0, 'ternary': None,
            'is_sarcasm': False, 'sarcasm_confidence': 0, 'sarcasm_reasons': [],
            'final_sentiment': None, 'flipped': False, 'reasoning': [],
            'details': [],
        }

        # Step 1: 标准化
        norm_text, hits = self.normalizer.normalize(text)
        clean = self.normalizer.normalize_clean(text)
        result['normalized'] = norm_text
        result['slang_hits'] = hits
        if hits:
            result['reasoning'].append(
                f"🔤 识别谐音/网络用语 {len(hits)} 处: "
                + ", ".join(f"{s}→{t}" for s, t in hits)
            )

        # Step 2: 模型预测
        pred, score = self._predict(clean)

        # Step 3: 讽刺检测
        pos_score = score if pred == 1 else 1 - score
        is_sarcasm, sarcasm_conf, sarcasm_reasons = self.detector.detect(text, pos_score)

        result['is_sarcasm'] = is_sarcasm
        result['sarcasm_confidence'] = sarcasm_conf
        result['sarcasm_reasons'] = sarcasm_reasons

        # Step 4: 综合判断
        if is_sarcasm and sarcasm_conf > 0.3:
            final_pred = 1 - pred
            result['flipped'] = True
            result['reasoning'].append(
                f"🔄 讽刺信号(置信度{sarcasm_conf:.0%})，情感翻转: "
                f"{'正面' if pred==1 else '负面'} → {'正面' if final_pred==1 else '负面'}"
            )
            for r in sarcasm_reasons:
                result['reasoning'].append(f"  └ {r}")
        else:
            final_pred = pred
            if sarcasm_reasons:
                result['reasoning'].append(
                    f"⚠️ 检测到弱讽刺信号({sarcasm_conf:.0%})，未触发翻转阈值"
                )

        result['pred'] = final_pred
        result['score'] = score
        result['final_sentiment'] = "正面" if final_pred == 1 else "负面"

        tc, sent, emoji, color = self._classify(final_pred, score)
        result['ternary'] = tc
        result['sentiment'] = sent
        result['emoji'] = emoji
        result['color'] = color
        result['confidence'] = score

        # 多模型对比
        for k, v in self.models.items():
            p, s = self._predict_single(clean, k, v)
            result['details'].append({'model': v.get('name', k), 'pred': p, 'score': s})

        return result

    def _predict_single(self, text, key, info):
        if 'pipe' in info:
            r = info['pipe'](text)[0]
            if isinstance(r, dict):
                label, score = r['label'], r['score']
            else:
                best = max(r, key=lambda x: x['score'])
                label, score = best['label'], best['score']
            pred = 1 if ('LABEL_1' in str(label) or label == '1') else 0
        else:
            import numpy as np
            cut = ' '.join(jieba.cut(text))
            X = info['vec'].transform([cut])
            pred = info['clf'].predict(X)[0]
            score = info['clf'].predict_proba(X)[0][pred] if hasattr(info['clf'], 'predict_proba') \
                    else 1/(1+np.exp(-abs(info['clf'].decision_function(X)[0])))
        return pred, float(score)

    def _predict(self, text):
        if not self.models:
            return 0, 0.5
        key = 'roberta' if 'roberta' in self.models else list(self.models.keys())[0]
        return self._predict_single(text, key, self.models[key])

    @staticmethod
    def _classify(pred, score):
        if pred == 1 and score > 0.65:
            return 2, "正面", "😊", "#22c55e"
        elif pred == 0 and score > 0.65:
            return 0, "负面", "😞", "#ef4444"
        else:
            return 1, "中性", "😐", "#f59e0b"


# ============================================================
#  CLI 测试
# ============================================================
if __name__ == '__main__':
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    normalizer = SlangNormalizer()
    detector = SarcasmDetector()

    print("=" * 70)
    print("     谐音标准化 + 讽刺检测 · 深化版测试")
    print("=" * 70)

    test_groups = {
        "🎮 贴吧/抽象话": [
            "典，这就是你们说的优质服务",
            "太孝了，这么差的质量还维护",
            "绷不住了，等了三个小时告诉我卖完了",
            "逆天，一个外卖送了三小时",
            "哈人，这价格也太离谱了",
        ],
        "🔤 谐音/网络语": [
            "针不戳，这个手机拍照绝绝子",
            "集美们避雷！这家店太下头了",
            "u1s1，这质量yyds，但价格属实刺客",
            "家人们谁懂啊，等了半天告诉我缺货",
            "city不city不知道，但服务态度真的窒息",
        ],
        "🎭 讽刺反话": [
            "这家店的服务真是太好了，等了两个小时才上菜",
            "多亏了快递，我的包裹一周后才到呢",
            "谢谢你们啊，把我手机摔得稀巴烂",
            "号称官方旗舰店，结果发的全是假货",
            "这就是你们说的优质服务？太有意思了",
        ],
        "🔀 混合 tricky": [
            "纯纯大冤种，花了两百块买了个破烂",
            "绝绝子的服务态度，等了半天告诉我没货，难绷",
            "咱就是说，这波操作属实栓Q，蚌埠住了",
            "遥遥领先的配送速度，三天了还没出仓库",
        ],
        "✅ 正常表达": [
            "味道不错价格实惠，下次还会来",
            "太难吃了，从来没吃过这么差的",
            "还行吧，无功无过，一般般",
        ],
    }

    for group_name, cases in test_groups.items():
        print(f"\n{'─'*70}")
        print(f"  {group_name}")
        print(f"{'─'*70}")
        for text in cases:
            # 标准化
            norm, hits = normalizer.normalize(text)
            slang_tag = f" 🔤×{len(hits)}" if hits else ""

            # 讽刺检测
            is_sar, conf, reasons = detector.detect(text, 0.75)
            sarc_tag = f" 🎭{conf:.0%}" if is_sar else ""

            print(f"\n  📝 {text}")
            if hits:
                for s, t in hits:
                    print(f"     🔤 {s} → {t}")
                print(f"     📄 标准化: {norm}")
            if is_sar:
                print(f"     🎭 讽刺 (置信度 {conf:.0%})")
                for r in reasons:
                    print(f"        └ {r}")
            if not hits and not is_sar:
                print(f"     ✅ 无特殊信号")
