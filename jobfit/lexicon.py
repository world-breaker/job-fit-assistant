"""技能词表：类别划分 + 归一化别名。

这是**规则兜底**与**关键词基线**共用的唯一事实源。两者的口径必须一致，
否则「基线 vs LLM」的对比就没有意义（基线弱不是因为口径不同，而是因为方法不同）。

设计要点：
- 词表按**类别**组织：语言 / 框架 / 工程 / 算法 / 业务；
- 每个技能给一组**别名**（大小写、简写、中英文写法变体）；
- 一个别名**只能**属于一个规范名。导入时做冲突检查，发现冲突直接报错——
  「以为词表是对的，实际同一个词被两条技能抢」会让抽取结果出现随机重复，
  比直接崩掉更难查（口径同 rag-mcp-server 的 fail fast）；
- 别名匹配一律**大小写不敏感**；对 C++ / C / JS 这类短词用**词边界**约束，
  避免 "C" 命中 "CUDA"、"CSS"。
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

#: 类别取值（与规格里的「语言/框架/工程/算法/业务」一一对应）
CATEGORIES = ("语言", "框架", "工程", "算法", "业务")

#: 技能词表：规范名 -> (类别, 别名列表)
SKILL_LEXICON: Dict[str, Tuple[str, List[str]]] = {
    # ---------------- 语言 ----------------
    "Python": ("语言", ["Python", "python3", "cpython"]),
    "Java": ("语言", ["Java", "JAVA", "java8", "java11", "java17"]),
    "C++": ("语言", ["C++", "c++", "cpp"]),
    "C": ("语言", ["C语言", "C/C++"]),
    "Go": ("语言", ["Golang", "Go语言"]),
    "Rust": ("语言", ["Rust", "rust-lang"]),
    "Scala": ("语言", ["Scala"]),
    "JavaScript": ("语言", ["JavaScript", "JS", "ES6", "TypeScript", "TS"]),
    "SQL": ("语言", ["SQL", "Hive SQL", "Spark SQL"]),
    "Shell": ("语言", ["Shell", "Bash", "Shell脚本"]),
    "Swift": ("语言", ["Swift"]),
    "Kotlin": ("语言", ["Kotlin"]),
    "MATLAB": ("语言", ["MATLAB", "Matlab"]),
    # ---------------- 框架 ----------------
    "PyTorch": ("框架", ["PyTorch", "pytorch"]),
    "TensorFlow": ("框架", ["TensorFlow", "TF2", "tensorflow"]),
    "Transformers": ("框架", ["Transformers", "transformers", "HuggingFace", "Hugging Face"]),
    "Scikit-learn": ("框架", ["scikit-learn", "sklearn", "Scikit-Learn"]),
    "XGBoost": ("框架", ["XGBoost", "xgboost", "LightGBM", "CatBoost"]),
    "Spring Boot": ("框架", ["Spring Boot", "SpringBoot"]),
    "MyBatis": ("框架", ["MyBatis", "Mybatis", "Hibernate"]),
    "Flask": ("框架", ["Flask"]),
    "FastAPI": ("框架", ["FastAPI", "Fast Api"]),
    "Vue": ("框架", ["Vue", "Vue.js", "Vue3", "React"]),
    "Docker": ("框架", ["Docker", "docker-compose"]),
    "Redis": ("框架", ["Redis"]),
    "MySQL": ("框架", ["MySQL", "Mysql"]),
    "PostgreSQL": ("框架", ["PostgreSQL", "Postgres"]),
    "Kafka": ("框架", ["Kafka"]),
    "Spark": ("框架", ["Spark", "PySpark"]),
    "Hadoop": ("框架", ["Hadoop", "HDFS"]),
    "Flink": ("框架", ["Flink"]),
    "Ray": ("框架", ["Ray"]),
    "Milvus": ("框架", ["Milvus", "Faiss", "Pinecone", "Weaviate"]),
    "vLLM": ("框架", ["vLLM", "vllm", "TensorRT-LLM", "SGLang"]),
    "DeepSpeed": ("框架", ["DeepSpeed", "deepspeed", "Megatron", "Megatron-LM"]),
    "PEFT/LoRA": ("框架", ["PEFT", "LoRA", "QLoRA"]),
    "LangChain": ("框架", ["LangChain", "LangGraph", "LlamaIndex", "Dify"]),
    "MLflow": ("框架", ["MLflow", "MLFlow", "SwanLab", "Weights & Biases", "W&B"]),
    "CUDA": ("框架", ["CUDA", "cudnn", "GPU 编程"]),
    "Elasticsearch": ("框架", ["Elasticsearch", "ES集群"]),
    "Maven": ("框架", ["Maven", "Gradle"]),
    "Nginx": ("框架", ["Nginx"]),
    "Prometheus": ("框架", ["Prometheus", "Grafana", "可观测"]),
    # ---------------- 工程 ----------------
    "Git": ("工程", ["Git", "github", "gitlab", "版本管理"]),
    "Linux": ("工程", ["Linux", "Unix"]),
    "CI/CD": ("工程", ["CI/CD", "Jenkins", "GitHub Actions", "GitLab CI", "持续集成"]),
    "Kubernetes": ("工程", ["Kubernetes", "K8s", "k8s"]),
    "微服务": ("工程", ["微服务", "微服务架构", "服务治理", "分布式系统"]),
    "RESTful API": ("工程", ["RESTful", "REST API", "RESTful API", "HTTP API", "接口设计"]),
    "分布式训练": ("工程", ["分布式训练", "多机多卡", "数据并行", "张量并行", "3D并行", "ZeRO"]),
    "模型部署": ("工程", ["模型部署", "推理部署", "模型上线"]),
    "模型蒸馏": ("工程", ["蒸馏", "模型蒸馏", "知识蒸馏"]),
    "模型量化": ("工程", ["量化", "INT8", "GPTQ", "AWQ", "FP8", "混合精度"]),
    "高并发": ("工程", ["高并发", "大流量", "QPS", "高性能", "低延迟", "吞吐"]),
    "高可靠": ("工程", ["高可靠", "可靠性", "容错"]),
    "性能优化": ("工程", ["性能优化", "优化性能", "压测", "profiling"]),
    "工程化": ("工程", ["工程化", "工程落地", "落地能力", "工程实践"]),
    "A/B实验": ("工程", ["A/B实验", "AB实验", "AB测试", "A/B测试"]),
    "服务端开发": ("工程", ["服务端开发", "后台开发", "后端开发"]),
    "服务端架构": ("工程", ["服务端架构", "后台架构"]),
    "架构设计": ("工程", ["架构设计", "系统架构设计", "技术架构", "平台架构设计"]),
    "任务编排": ("工程", ["任务编排", "工作流编排", "调度编排"]),
    "模型接入": ("工程", ["模型接入", "模型接入服务"]),
    "部署发布": ("工程", ["部署发布", "发布上线"]),
    "部署": ("工程", ["部署", "上线部署"]),
    "代码评审": ("工程", ["代码评审", "Code Review", "code review"]),
    "数据库设计": ("工程", ["数据库设计", "表结构设计", "数据建模"]),
    "消息队列": ("框架", ["消息队列", "MQ", "RocketMQ", "Pulsar"]),
    "Spring Cloud": ("框架", ["Spring Cloud", "SpringCloud", "微服务框架"]),
    "DevOps": ("工程", ["DevOps", "Devops", "devops", "CI/CD 平台"]),
    # ---------------- 算法 ----------------
    "Word2Vec": ("算法", ["Word2Vec", "word2vec", "CBOW", "Skip-gram", "Skipgram", "负采样"]),
    "BERT": ("算法", ["BERT", "RoBERTa", "ERNIE"]),
    "LLM微调": ("算法", ["微调", "SFT", "LoRA微调", "指令微调", "全参微调", "Fine-tuning", "finetune"]),
    "DPO": ("算法", ["DPO", "RLHF", "PPO", "GRPO", "强化学习", "奖励模型", "RLVR", "对齐算法"]),
    "后训练": ("算法", ["后训练", "Post-training"]),
    "Transformer": ("算法", ["Transformer", "Attention", "自注意力", "Mamba", "MoE"]),
    "RAG": ("算法", ["RAG", "检索增强", "知识库问答"]),
    "Agent": ("算法", ["Agent", "智能体", "Multi-Agent", "多智能体", "工具调用", "Tool Calling", "Agent 编排", "Agentic"]),
    "Prompt工程": ("算法", ["Prompt 设计", "Prompt设计", "Prompt工程", "Prompt 优化", "Prompt优化", "PE（提示工程）", "提示工程"]),
    "大模型多阶段训练": ("算法", ["大模型多阶段训练", "多阶段训练", "预训练、"]),
    "大模型推理": ("算法", ["推理优化", "推理加速", "KV Cache", "投机解码", "prefill", "低延迟推理", "端侧推理", "推理服务"]),
    "AI应用开发": ("算法", ["AI 应用开发", "AI应用开发", "大模型应用开发", "AI 应用研发", "AI应用研发"]),
    "大模型应用落地": ("业务", ["大模型应用落地", "大模型落地", "应用落地经验"]),
    "AI开发工具": ("工程", ["AI 开发工具", "AI开发工具", "AI 辅助开发", "AI辅助开发", "Copilot", "Codex"]),
    "文本分类": ("算法", ["文本分类", "分类任务", "意图识别", "情感分析"]),
    "序列标注": ("算法", ["序列标注", "NER", "命名实体识别", "分词", "词性标注"]),
    "机器学习": ("算法", ["机器学习", "Machine Learning", "深度学习", "ML"]),
    "NLP": ("算法", ["NLP", "自然语言处理", "nlp", "文本理解"]),
    "NumPy": ("框架", ["NumPy", "numpy", "Numpy"]),
    "数据结构与算法": ("算法", ["数据结构与算法", "数据结构", "算法功底", "算法基础"]),
    "模型评估": ("算法", ["模型评估", "模型评价", "效果评估", "评估方法"]),
    "大模型评测": ("业务", ["大模型评测", "模型评测", "评测体系", "评测基准", "LLM-as-a-Judge", "MMLU", "SWE-bench", "HumanEval", "AgentBench"]),
    "评测集构建": ("业务", ["评测集构建", "评测集建设", "评测集", "数据集生产链路", "评测数据"]),
    "评测指标体系": ("业务", ["评测指标体系", "指标体系", "多维度指标", "评价体系"]),
    "自动化评测": ("工程", ["自动化评测", "评测工程化", "评测工具链", "评测流水线", "自动打分", "批量测试"]),
    "多模态理解": ("业务", ["多模态理解", "音视频理解", "跨模态理解"]),
    "图像超分辨率": ("算法", ["超分辨率", "超分", "FSRCNN", "SRGAN", "ESRGAN"]),
    "三维重建": ("算法", ["三维重建", "网格重构", "点云", "三维网格", "Mesh", "Open3D", "NeRF", "3DGS"]),
    "向量检索": ("算法", ["向量检索", "相似度检索", "ANN", "余弦相似度", "embedding", "Embedding", "向量数据库"]),
    # ---------------- 业务 ----------------
    "推荐系统": ("业务", ["推荐系统", "推荐算法", "CTR", "召回", "排序模型"]),
    "搜索": ("业务", ["搜索引擎", "搜索系统", "召回排序", "BM25"]),
    "风控": ("业务", ["风控", "反作弊", "内容安全", "审核"]),
    "金融科技": ("业务", ["金融", "风控模型", "信贷", "量化交易", "支付"]),
    "医疗AI": ("业务", ["医疗", "医学影像", "医疗AI"]),
    "教育AI": ("业务", ["教育", "教学", "题库"]),
    "电商": ("业务", ["电商", "交易", "商品", "购物车", "订单系统"]),
    "社交": ("业务", ["社交", "IM", "即时通讯", "小程序"]),
    "游戏": ("业务", ["游戏", "游戏引擎", "Unity", "关卡", "副本"]),
    "数据标注": ("业务", ["数据标注", "标注", "数据清洗", "语料", "数据治理", "训练数据合成", "语料处理", "语料清洗"]),
    "AIGC": ("业务", ["AIGC", "多模态", "文生图", "图像生成", "数字人", "语音合成"]),
}


def _build_reverse_index() -> Dict[str, str]:
    """别名 -> 规范名。发现同一个别名被两个技能占用就直接报错。"""
    reverse: Dict[str, str] = {}
    owner: Dict[str, str] = {}
    for canon, (_cat, aliases) in SKILL_LEXICON.items():
        for alias in [canon] + list(aliases):
            key = alias.strip().lower()
            if not key:
                continue
            if key in reverse and reverse[key] != canon:
                raise ValueError(
                    f"技能词表别名冲突：别名 {alias!r} 同时属于 {reverse[key]!r} 与 {canon!r}。"
                    " 一个别名只能归属一个规范名，请修正词表。"
                )
            reverse[key] = canon
            owner[key] = canon
    return reverse


#: 别名（小写）-> 规范名
ALIAS_TO_CANON: Dict[str, str] = _build_reverse_index()

#: 规范名 -> 类别
SKILL_CATEGORY: Dict[str, str] = {name: cat for name, (cat, _) in SKILL_LEXICON.items()}

#: 规范名 -> 别名列表（规范名自身也在首位）
SKILL_ALIASES: Dict[str, List[str]] = {
    name: list(dict.fromkeys([name] + list(aliases)))
    for name, (_cat, aliases) in SKILL_LEXICON.items()
}

#: 需要词边界约束的别名（短且容易误命中）
_WORD_BOUNDARY_ALIASES = {"c++", "c", "js", "ts", "go", "sql", "flink", "ray", "k8s", "es6"}

#: **歧义排除**：某些别名在特定上下文里是另一个意思，命中了也不算该技能。
#:
#: 例：「超参网格**搜索**」里的「搜索」是 grid search（超参搜索），
#:     不是 JD 要求的「搜索算法/搜索引擎」——按字面匹配会造出**假绿**
#:     （说候选人具备「搜索」技能，实际不具备）。
#:     命中这些别名时，若**前文**出现列表里的词，则该次命中作废。
#:
#: 键是别名（小写），值是作废标记词列表。
AMBIGUITY_EXCLUSIONS: Dict[str, List[str]] = {
    # 「超参网格搜索」= grid search，不是「搜索算法/搜索引擎」
    "搜索": ["网格", "超参", "参数", "暴力", "二分", "广度", "深度", "线性", "回溯", "树"],
    "检索": ["网格", "暴力", "顺序"],
    # 「量化对比实验」= 定量对比，不是「模型量化（INT8/GPTQ…）」
    "量化": ["对比", "实验", "指标", "评估", "分析", "对比实验", "关系"],
    "INT8": ["对比", "实验"],
    # 「蒸馏」在「蒸馏对比」里也是定量含义（这里先不设，避免误杀真实的模型蒸馏）
}
#: 上表中值为空列表的条目不生效（占位说明）
AMBIGUITY_EXCLUSIONS = {k: v for k, v in AMBIGUITY_EXCLUSIONS.items() if v}


def canonical_name(alias: str) -> str | None:
    """把任意别名映射到规范技能名；不认识返回 None。"""
    return ALIAS_TO_CANON.get(alias.strip().lower())


def all_aliases() -> List[Tuple[str, str]]:
    """返回 [(规范名, 别名), ...]，供构建匹配器使用。"""
    out: List[Tuple[str, str]] = []
    for canon, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            out.append((canon, alias))
    return out


def _alias_pattern(alias: str) -> re.Pattern:
    """为别名构造正则：长别名直接子串匹配，短别名加词边界。"""
    escaped = re.escape(alias)
    if alias.strip().lower() in _WORD_BOUNDARY_ALIASES:
        # \b 对 "C++" 这类以非单词字符结尾的别名不可靠，改用「前后不是字母数字/加号」
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9+])", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def is_ambiguous_hit(text: str, start: int, end: int, alias: str) -> bool:
    """判断这次命中是否因为**词义歧义**而应当作废。

    判据：别名有歧义排除规则，且命中的**前文或后文**（同句内、命中位置附近）
    出现了作废标记词。

    例 1：「支持动态负采样、多窗口对比、多轮超参网格搜索」里的「搜索」，
          前文有「超参」「网格」→ 是「超参网格搜索」，不是「搜索算法」→ 作废。
    例 2：「设计 5 类工业纹理的量化对比实验」里的「量化」，
          后文有「对比」→ 是「定量对比实验」，不是「模型量化（INT8/GPTQ）」→ 作废。

    ★ 为什么前后都看：中文里歧义词经常跟在修饰语**后面**
    （「量化**对比**实验」），只看前文会漏掉。
    """
    exclusions = AMBIGUITY_EXCLUSIONS.get(alias.strip().lower())
    if not exclusions:
        return False
    # 取命中所在句子
    left_bound = max(
        text.rfind("。", 0, start), text.rfind("\n", 0, start),
        text.rfind("；", 0, start), text.rfind(";", 0, start),
    )
    right_bound = len(text)
    for i in range(end, len(text)):
        if text[i] in "。；;\n":
            right_bound = i
            break
    prefix = text[left_bound + 1 : start]
    suffix = text[end:right_bound]
    return any(marker in prefix for marker in exclusions) or any(
        marker in suffix for marker in exclusions
    )


#: 预编译匹配器：[(规范名, 别名, 正则), ...]
MATCHERS: List[Tuple[str, str, re.Pattern]] = [
    (canon, alias, _alias_pattern(alias)) for canon, alias in all_aliases()
]


def category_of(canonical: str) -> str:
    """规范名的类别；未知技能归到「算法」（最保守的选择，见 matching.py 的说明）。"""
    return SKILL_CATEGORY.get(canonical, "算法")
