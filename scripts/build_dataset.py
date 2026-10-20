"""构建评测数据集：12 份 JD（带人工标注）+ 4 份脱敏简历。

★ 数据来源分两类（规格要求 ≥6 份来自公开可查的真实 JD 文本）：
- **公开真实 JD**（9 份）：来自腾讯招聘官网公开接口，附来源链接与抓取日期。
  抓取脚本：scripts/fetch_jds.py；原始抓取结果：data/jds/raw/tencent_raw.json。
- **自写变体**（3 份）：基于真实 JD 的结构与内容改写而成（改了岗位方向与措辞），
  用来补齐评测集在「Java 后端 / NLP 算法 / 大模型评测」三个方向上的样本量。
  ★ 明确标注 `source_type: "authored_variant"`，不冒充真实 JD。

★ 人工标注口径（写进 data/jds/ANNOTATION.md）：
1. **什么算一条要求**：JD 里明确写出的「技能/技术/工具/方法/能力」，
   且是**该岗位要招人具备**的东西。排除：学历年限、软素质（沟通/责任心）、
   岗位职责里的业务描述、薪资福利。
2. **重要度判定**：
   - 「必须」= 原文出现 精通/熟练掌握/扎实/深入理解/具备…经验/至少…
     等硬性表述，或写在「任职要求」的无条件句里；
   - 「加分」= 原文出现 优先/加分/更佳/有兴趣/有…经验者优先。
   - 一句话里两者都有（「熟悉 X，有 Y 经验者优先」）→ 判「必须」（硬性条件在前）。
3. **技能名归一化**：用 jobfit.normalize 的口径（词表映射 + 复合短语拆分），
   这样 LLM 抽取、规则抽取、人工标注三方能对齐。
4. **原文证据**：必须是 JD 原文的**逐字片段**（脚本会校验，不匹配直接报错）。

用法：
    python scripts/build_dataset.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jobfit.normalize import normalize_key  # noqa: E402

RAW_PATH = ROOT / "data" / "jds" / "raw" / "tencent_raw.json"
OUT_DIR = ROOT / "data" / "jds"

#: 人工标注：(post_id, [(技能, 类别, 重要度, 原文证据片段)])
#:
#: 证据片段写 JD 里**逐字**的一段短文本；build 脚本会在原文里定位到完整句子，
#: 定位不到就报错——避免标注与原文脱节。
ANNOTATIONS: Dict[str, List[Tuple[str, str, str, str]]] = {
    # ---------------------------------------------------------------- 公开真实 JD
    "2008789833664847872": [
        ("后台开发", "工程", "必须", "具备良好的后台开发能力"),
        ("服务端架构", "工程", "必须", "了解常用服务端架构、存储与中间件技术"),
        ("Prompt设计", "算法", "必须", "熟悉 Prompt 设计"),
        ("RAG", "算法", "必须", "RAG 检索增强"),
        ("Agent", "算法", "必须", "Agent 编排"),
        ("工具调用", "算法", "必须", "工具调用等常见技术"),
        ("大模型应用开发", "算法", "必须", "了解大模型应用开发"),
        ("任务编排", "工程", "必须", "任务编排、上下文管理与记忆机制"),
        ("模型接入", "工程", "必须", "模型接入"),
        ("部署发布", "工程", "必须", "部署发布、监控告警、限流降级"),
        ("性能优化", "工程", "必须", "性能与成本优化"),
        ("AIGC", "业务", "必须", "AIGC 工具"),
        ("游戏", "业务", "加分", "对 AI 与游戏结合有热情"),
    ],
    "2108081571104997376": [
        ("Python", "语言", "必须", "熟练掌握 Python、Go 或 C++ 中至少一种语言"),
        ("Go", "语言", "必须", "熟练掌握 Python、Go 或 C++ 中至少一种语言"),
        ("C++", "语言", "必须", "熟练掌握 Python、Go 或 C++ 中至少一种语言"),
        ("服务端开发", "工程", "必须", "具备 3 年以上服务端或 AI 应用开发经验"),
        ("AI应用开发", "工程", "必须", "具备 3 年以上服务端或 AI 应用开发经验"),
        ("DevOps", "工程", "必须", "熟悉 DevOps 领域 AI 产品与工具开发"),
        ("大模型应用落地", "业务", "加分", "有大模型应用落地经验者优先"),
        ("架构设计", "工程", "必须", "主导技术方案设计与代码评审"),
        ("代码评审", "工程", "必须", "主导技术方案设计与代码评审"),
    ],
    "2098594908700061696": [
        ("Java", "语言", "必须", "具备扎实的 Java 基础和良好的工程实践"),
        ("Spring Boot", "框架", "必须", "熟悉 Spring Boot、Spring Cloud、MySQL、Redis、消息队列等常用技术"),
        ("Spring Cloud", "框架", "必须", "熟悉 Spring Boot、Spring Cloud、MySQL、Redis、消息队列等常用技术"),
        ("MySQL", "框架", "必须", "熟悉 Spring Boot、Spring Cloud、MySQL、Redis、消息队列等常用技术"),
        ("Redis", "框架", "必须", "熟悉 Spring Boot、Spring Cloud、MySQL、Redis、消息队列等常用技术"),
        ("消息队列", "框架", "必须", "消息队列等常用技术"),
        ("数据库设计", "工程", "必须", "具备数据库设计、性能优化及相关项目经验"),
        ("性能优化", "工程", "必须", "具备数据库设计、性能优化及相关项目经验"),
        ("分布式系统", "工程", "必须", "具备分布式系统设计和生产实践能力"),
        ("微服务", "工程", "必须", "具备分布式系统设计和生产实践能力"),
        ("电商", "业务", "必须", "具备电商或其他复杂交易系统的分析与建模能力"),
        ("AI开发工具", "工程", "必须", "能够熟练使用 AI 开发工具"),
    ],
    "2089960263330742272": [
        ("Java", "语言", "必须", "具备扎实的 Java 基础和良好的工程实践"),
        ("Spring Boot", "框架", "必须", "熟悉 Spring Boot、Spring Cloud、MySQL、Redis、消息队列等常用技术"),
        ("Spring Cloud", "框架", "必须", "熟悉 Spring Boot、Spring Cloud、MySQL、Redis、消息队列等常用技术"),
        ("MySQL", "框架", "必须", "熟悉 Spring Boot、Spring Cloud、MySQL、Redis、消息队列等常用技术"),
        ("Redis", "框架", "必须", "熟悉 Spring Boot、Spring Cloud、MySQL、Redis、消息队列等常用技术"),
        ("消息队列", "框架", "必须", "消息队列等常用技术"),
        ("数据库设计", "工程", "必须", "具备数据库设计、性能优化及相关项目经验"),
        ("性能优化", "工程", "必须", "具备数据库设计、性能优化及相关项目经验"),
        ("分布式系统", "工程", "必须", "具备分布式系统设计和生产实践能力"),
        ("微服务", "工程", "必须", "具备分布式系统设计和生产实践能力"),
        ("电商", "业务", "必须", "具备电商或其他复杂交易系统的分析与建模能力"),
        ("AI开发工具", "工程", "必须", "能够熟练使用 AI 开发工具"),
    ],
    "2092134938500579328": [
        ("后训练", "算法", "必须", "具备扎实的后训练、NLP，熟练掌握大模型多阶段训练、推理加速、排序推荐等核心算法原理"),
        ("NLP", "算法", "必须", "具备扎实的后训练、NLP"),
        ("大模型多阶段训练", "算法", "必须", "熟练掌握大模型多阶段训练、推理加速、排序推荐等核心算法原理"),
        ("推理加速", "算法", "必须", "熟练掌握大模型多阶段训练、推理加速、排序推荐等核心算法原理"),
        ("LLM微调", "算法", "必须", "熟练掌握微调、量化、蒸馏、Prompt工程、RAG等技术"),
        ("模型量化", "算法", "必须", "熟练掌握微调、量化、蒸馏、Prompt工程、RAG等技术"),
        ("模型蒸馏", "算法", "必须", "熟练掌握微调、量化、蒸馏、Prompt工程、RAG等技术"),
        ("Prompt工程", "算法", "必须", "熟练掌握微调、量化、蒸馏、Prompt工程、RAG等技术"),
        ("RAG", "算法", "必须", "熟练掌握微调、量化、蒸馏、Prompt工程、RAG等技术"),
        ("大模型推理", "算法", "必须", "具备大模型轻量化、低延迟推理优化实战经验"),
        ("搜索", "业务", "加分", "有输入法、键盘、文本预测、搜索推荐等相关场景算法优化经验者优先"),
        ("推荐系统", "业务", "加分", "搜索推荐等相关场景算法优化经验者优先"),
        ("数据标注", "业务", "必须", "包含数据挖掘分析、bad case归因"),
        ("A/B实验", "工程", "必须", "从需求拆解、实验验证、灰度上线到复盘迭代的全流程"),
    ],
    "2082710394723545088": [
        ("Python", "语言", "必须", "熟练使用 Python，具备良好的工程能力"),
        ("Transformer", "算法", "必须", "深入理解 Transformer 架构及主流大模型训练框架"),
        ("PyTorch", "框架", "必须", "深入理解 Transformer 架构及主流大模型训练框架"),
        ("LLM微调", "算法", "必须", "熟悉 SFT 与 RLVR 训练流程"),
        ("RLVR", "算法", "必须", "熟悉 SFT 与 RLVR 训练流程"),
        ("DPO", "算法", "必须", "理解奖励信号设计、强化学习算法调优与训练稳定性问题"),
        ("数据标注", "业务", "必须", "能进行训练数据合成、清洗、价值评估与 badcase 归因"),
        ("AIGC", "业务", "加分", "有大规模代码语料处理经验者优先"),
    ],
    "2094265700456771584": [
        ("Python", "语言", "必须", "精通Python程序开发，有良好的编码风格及创新优化意识"),
        ("大模型评测", "业务", "必须", "有大模型训练或模型评估经验，熟悉大模型评测方法"),
        ("评测集构建", "业务", "必须", "包含模型的评测体系建设、评测集构建"),
        ("大模型评测", "业务", "必须", "跟踪业界评测基准动态"),
        ("vLLM", "框架", "加分", "对推理框架（如sglang、vllm等）有经验的优先"),
        ("SGLang", "框架", "加分", "对推理框架（如sglang、vllm等）有经验的优先"),
        ("工程化", "工程", "必须", "推动评测工程化和无人化"),
        ("Prompt工程", "算法", "加分", "PE（提示工程）实践经验者优先"),
    ],
    "1915233782491881472": [
        ("大模型评测", "业务", "必须", "通用AI大模型评测基准的构建"),
        ("多模态理解", "业务", "必须", "建立覆盖文生文、多模态理解、多模态（音视频/3D/图/视频生成）生成等模态的评测基准"),
        ("AIGC", "业务", "必须", "建立覆盖文生文、多模态理解、多模态（音视频/3D/图/视频生成）生成等模态的评测基准"),
        ("评测指标体系", "业务", "必须", "设计全面、准确的多维度指标"),
        ("自动化评测", "工程", "必须", "构建自动化评测工具链"),
        ("数据标注", "业务", "必须", "构建领域增强型评测数据集生产链路"),
        ("NLP", "算法", "必须", "有主流深度学习模型的项目(CV/NLP/推荐/广告/RL等均可）实践落地经验优先"),
        ("模型评估", "算法", "必须", "有大模型调优化应用、评测经历者优先"),
        ("机器学习", "算法", "必须", "扎实的机器学习基础，能够熟练应用常用的机器学习模型解决实际的业务问题"),
    ],
    "2095418906612248576": [
        ("LLM微调", "算法", "必须", "熟练掌握 SFT、RL 等模型训练方法，有丰富的模型调优和部署经验"),
        ("DPO", "算法", "必须", "熟练掌握 SFT、RL 等模型训练方法"),
        ("PyTorch", "框架", "必须", "熟悉常用算法和框架（如 PyTorch、TensorFlow ）"),
        ("TensorFlow", "框架", "必须", "熟悉常用算法和框架（如 PyTorch、TensorFlow ）"),
        ("工具调用", "算法", "必须", "提升大模型在工具调用、对话理解的表现"),
        ("模型评估", "算法", "必须", "建立评估体系，多维度评测量化智能体的执行质量"),
        ("部署", "工程", "必须", "有丰富的模型调优和部署经验"),
    ],
    # ---------------------------------------------------------------- 自写变体
    "authored_java_fintech": [
        ("Java", "语言", "必须", "熟练掌握 Java"),
        ("Spring Boot", "框架", "必须", "熟练使用 Spring Boot 与 Spring Cloud 开发微服务"),
        ("MySQL", "框架", "必须", "熟悉 MySQL 索引优化与分库分表"),
        ("Redis", "框架", "必须", "了解 Redis 缓存设计与一致性"),
        ("Kafka", "框架", "必须", "有 Kafka 消息队列实践"),
        ("分布式系统", "工程", "必须", "有高并发分布式系统设计经验"),
        ("微服务", "工程", "必须", "熟练使用 Spring Boot 与 Spring Cloud 开发微服务"),
        ("高并发", "工程", "必须", "有高并发分布式系统设计经验"),
        ("金融科技", "业务", "必须", "有金融支付或风控相关系统开发经验者优先"),
        ("风控", "业务", "加分", "有金融支付或风控相关系统开发经验者优先"),
    ],
    "authored_nlp_classification": [
        ("Python", "语言", "必须", "熟练使用 Python 进行算法开发"),
        ("PyTorch", "框架", "必须", "熟悉 PyTorch 训练与微调流程"),
        ("BERT", "算法", "必须", "有 BERT、RoBERTa 等预训练模型微调经验"),
        ("LLM微调", "算法", "必须", "有大模型微调（SFT/LoRA）实践经验"),
        ("PEFT/LoRA", "框架", "必须", "有大模型微调（SFT/LoRA）实践经验"),
        ("文本分类", "算法", "必须", "有文本分类、意图识别等任务的经验"),
        ("NLP", "算法", "必须", "熟悉 NLP 基础任务与数据处理流程"),
        ("数据标注", "业务", "必须", "有语料清洗与标注经验"),
        ("模型评估", "算法", "必须", "建立分类任务的 P/R/F1 评估体系"),
    ],
    "authored_eval_platform": [
        ("Python", "语言", "必须", "精通 Python 编写评测工具与平台"),
        ("大模型评测", "业务", "必须", "有大模型评测体系从 0 到 1 的建设经验"),
        ("评测集构建", "业务", "必须", "能独立构建评测集与标注规范"),
        ("LLM-as-a-Judge", "算法", "必须", "熟悉 LLM-as-a-Judge 等裁判模型的评测方法"),
        ("指标体系", "业务", "必须", "能设计并实现多维度指标体系"),
        ("工程化", "工程", "必须", "有评测工程化、平台化落地经验"),
        ("MLflow", "框架", "加分", "有 MLflow 等实验管理平台使用经验者优先"),
        ("A/B实验", "工程", "加分", "熟悉 A/B 实验与效果归因分析者优先"),
    ],
}

#: 自写变体 JD 的正文（用于「其余为自写变体」那部分）
AUTHORED_JDS: Dict[str, Dict[str, str]] = {
    "authored_java_fintech": {
        "title": "金融科技-Java后端开发工程师（自写变体）",
        "raw_text": (
            "1.负责支付清结算核心系统的设计、开发与上线，支撑日均千万级交易；\n"
            "2.参与交易链路的风控能力建设，包含限额、鉴权、异常交易检测；\n"
            "3.负责系统性能与稳定性治理，推动核心链路持续演进。\n"
            "1.熟练掌握 Java，熟悉 JVM 内存模型与并发编程；\n"
            "2.熟练使用 Spring Boot 与 Spring Cloud 开发微服务，理解服务治理与分布式事务；\n"
            "3.熟悉 MySQL 索引优化与分库分表，了解 Redis 缓存设计与一致性；\n"
            "4.有 Kafka 消息队列实践，了解消息可靠投递与消费幂等；\n"
            "5.有高并发分布式系统设计经验，能独立定位线上问题；\n"
            "6.有金融支付或风控相关系统开发经验者优先。"
        ),
    },
    "authored_nlp_classification": {
        "title": "文本智能-NLP算法工程师（自写变体）",
        "raw_text": (
            "1.负责文本分类与意图识别模型的训练、微调与上线；\n"
            "2.负责语料清洗、标注规范制定与训练数据构造；\n"
            "3.建立模型效果评估体系，持续迭代线上指标。\n"
            "1.熟练使用 Python 进行算法开发，熟悉 PyTorch 训练与微调流程；\n"
            "2.有 BERT、RoBERTa 等预训练模型微调经验，理解其结构与训练目标；\n"
            "3.有大模型微调（SFT/LoRA）实践经验，理解指令数据构造；\n"
            "4.熟悉 NLP 基础任务与数据处理流程，有文本分类、意图识别等任务的经验；\n"
            "5.有语料清洗与标注经验，能制定可执行的标注规范；\n"
            "6.建立分类任务的 P/R/F1 评估体系，能做错误案例归因分析。"
        ),
    },
    "authored_eval_platform": {
        "title": "大模型效果评测-评测平台工程师（自写变体）",
        "raw_text": (
            "1.负责大模型效果评测平台的架构设计与开发，支撑评测任务批量执行与结果可视化；\n"
            "2.建设评测集与标注规范，组织标注质量抽检；\n"
            "3.建设自动化评测流水线，打通数据、模型、指标与报告链路。\n"
            "1.精通 Python 编写评测工具与平台，有良好的工程实践；\n"
            "2.有大模型评测体系从 0 到 1 的建设经验，能独立设计指标口径；\n"
            "3.能独立构建评测集与标注规范，理解标注一致性的度量方法；\n"
            "4.熟悉 LLM-as-a-Judge 等裁判模型的评测方法，理解裁判偏差与校准手段；\n"
            "5.能设计并实现多维度指标体系，如准确率、召回率、通过率与稳定性指标；\n"
            "6.有评测工程化、平台化落地经验；\n"
            "7.有 MLflow 等实验管理平台使用经验者优先；\n"
            "8.熟悉 A/B 实验与效果归因分析者优先。"
        ),
    },
}

#: 每个 JD 的「期望推荐哪版简历」（端到端评测的标签）
#:
#: ★ 标签口径：按 JD 的**岗位族**标注（不是按「哪个人更适合」——
#:   那需要人的判断，不是可复现的评测）。四版简历对应四个岗位族：
#:   ai_engineering（AI工程）/ java_backend（Java开发）/ nlp_algorithm（NLP算法）
#:   / eval_posttrain（评测后训练）。
#:   标注依据：JD 的核心职责与主要技术栈落在哪个岗位族的主线上。
EXPECTED_RESUME: Dict[str, str] = {
    "2008789833664847872": "ai_engineering",
    "2108081571104997376": "ai_engineering",
    "2098594908700061696": "java_backend",
    "2089960263330742272": "java_backend",
    "2092134938500579328": "nlp_algorithm",
    "2082710394723545088": "eval_posttrain",
    "2094265700456771584": "eval_posttrain",
    "1915233782491881472": "eval_posttrain",
    "2095418906612248576": "eval_posttrain",
    "authored_java_fintech": "java_backend",
    "authored_nlp_classification": "nlp_algorithm",
    "authored_eval_platform": "eval_posttrain",
}

#: 简历文件 → 岗位族
RESUME_FAMILY: Dict[str, str] = {
    "简历202607-AI工程.md": "ai_engineering",
    "简历202607-Java开发.md": "java_backend",
    "简历202607-NLP算法.md": "nlp_algorithm",
    "简历202607-评测后训练-有联.md": "eval_posttrain",
}

#: 标注口径说明（写进 data/jds/ANNOTATION.md）
ANNOTATION_RUBRIC = {
    "什么算一条要求": (
        "JD 里明确写出的「技能/技术/工具/方法/能力」，且是该岗位要招人具备的东西。"
        "排除：学历与年限要求、软素质（沟通/责任心/学习能力）、"
        "纯岗位职责里的业务描述、薪酬福利。"
    ),
    "重要度判定": {
        "必须": "原文出现 精通/熟练掌握/扎实/深入理解/具备…经验/至少… 等硬性表述，"
        "或写在任职要求里的无条件句。",
        "加分": "原文出现 优先/加分/更佳/有兴趣/…者优先。",
        "两者都有": "「熟悉 X，有 Y 经验者优先」这类 → 判「必须」（硬性条件在前）。",
    },
    "技能名归一化": (
        "用 jobfit.normalize 的口径：先查技能词表映射到规范名，"
        "复合短语（如「高并发、高可靠海量服务开发」）拆成单个技术名词。"
        "这样 LLM 抽取、规则抽取、人工标注三方能对齐到同一集合上。"
    ),
    "原文证据": "必须是 JD 原文的逐字片段；build 脚本会校验，定位不到直接报错。",
    "标注者": "本项目作者（一人标注，未做多人交叉验证——这是本评测的已知边界）",
    "已知偏差": (
        "单人标注、无第二标注者复核；「重要度」的边界案例（如「熟悉 X，有 Y 经验者优先」）"
        "按上文规则统一判为「必须」。评测报告里把这写成了边界，不当结论用。"
    ),
}


def _find_sentence(text: str, anchor: str) -> str:
    """在原文中定位包含 anchor 的完整句子。

    找不到就抛错——**标注与原文脱节是必须炸出来的错误**，
    不能悄悄产出一份对不上的数据集。
    """
    idx = text.find(anchor)
    if idx < 0:
        raise ValueError(f"标注证据在原文中找不到：{anchor!r}")
    # 向前找句首
    start = 0
    for i in range(idx - 1, -1, -1):
        ch = text[i]
        if ch in "。；;\n":
            start = i + 1
            break
    # 向后找句末
    end = len(text)
    for i in range(idx + len(anchor), len(text)):
        if text[i] in "。；;\n":
            end = i + 1
            break
    return text[start:end].strip()


def build() -> int:
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    by_id = {r["post_id"]: r for r in raw["records"]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built = 0
    index: List[Dict[str, Any]] = []

    for jd_id, annotations in ANNOTATIONS.items():
        if jd_id in AUTHORED_JDS:
            meta = AUTHORED_JDS[jd_id]
            text = meta["raw_text"]
            source_type = "authored_variant"
            source_url = ""
            source_site = "自写变体（本项目作者撰写，非真实 JD）"
            fetched_at = ""
            title = meta["title"]
            location = ""
            category = "技术"
        else:
            record = by_id.get(jd_id)
            if record is None:
                print(f"[跳过] 原始数据里没有 post_id={jd_id}", file=sys.stderr)
                continue
            text = record["raw_text"]
            source_type = "public_real"
            source_url = record["source_url"]
            source_site = record["source_site"]
            fetched_at = record["fetched_at"]
            title = record["title"]
            location = record.get("location", "")
            category = record.get("category", "")

        requirements: List[Dict[str, Any]] = []
        for skill, category_name, importance, anchor in annotations:
            evidence = _find_sentence(text, anchor)
            requirements.append(
                {
                    "skill": skill,
                    "category": category_name,
                    "importance": importance,
                    "evidence": evidence,
                    "anchor": anchor,
                }
            )

        payload = {
            "id": jd_id,
            "title": title,
            "source_type": source_type,
            "source_site": source_site,
            "source_url": source_url,
            "fetched_at": fetched_at,
            "location": location,
            "category": category,
            "expected_resume": EXPECTED_RESUME.get(jd_id, ""),
            "raw_text": text,
            "annotations": {
                "rubric": ANNOTATION_RUBRIC,
                "requirements": requirements,
            },
        }
        target = OUT_DIR / f"{jd_id}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        built += 1
        index.append(
            {
                "id": jd_id,
                "title": title,
                "source_type": source_type,
                "source_url": source_url,
                "fetched_at": fetched_at,
                "expected_resume": EXPECTED_RESUME.get(jd_id, ""),
                "requirements": len(requirements),
            }
        )
        print(f"[完成] {jd_id:22s} {title[:32]:34s} {source_type:18s} {len(requirements)} 条标注")

    # 索引文件（便于人快速核对数据集构成）
    (OUT_DIR / "index.json").write_text(
        json.dumps(
            {
                "count": built,
                "public_real": sum(1 for i in index if i["source_type"] == "public_real"),
                "authored_variant": sum(1 for i in index if i["source_type"] == "authored_variant"),
                "resume_family": RESUME_FAMILY,
                "items": index,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n共 {built} 份 JD（真实 {sum(1 for i in index if i['source_type'] == 'public_real')} / "
          f"自写变体 {sum(1 for i in index if i['source_type'] == 'authored_variant')}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
