# job-fit-assistant（求职助手）

> 把一份 JD 和一份简历变成一份**能直接用**的求职报告：
> 抽技能要求 → 解析简历 → 逐条匹配（带简历原句证据）→ 差距与优先级建议 →
> 简历改写建议 → 预测面试题。

这不是「再造一个地基」，而是一个**真能用的应用**：
作者本人正在求职，这个工具跑在自己的求职流程里（见 [端到端 Demo](#端到端-demo)）。

---

## 目录

- [它解决什么问题](#它解决什么问题)
- [六步功能](#六步功能)
- [快速开始](#快速开始)
- [评测结果](#评测结果)
- [与三个地基仓库的组合](#与三个地基仓库的组合)
- [目录约定](#目录约定)
- [隐私与脱敏](#隐私与脱敏)
- [评测数据集](#评测数据集)
- [LLM 与费用](#llm-与费用)
- [诚实边界](#诚实边界)
- [开发](#开发)

---

## 它解决什么问题

投简历前最大的浪费是**投错方向**：简历上写的东西和 JD 要的东西对不上，
投出去石沉大海，还以为是自己不够努力。

已有的 AI 简历工具大多停在「帮你润色措辞」，但**不告诉你该投哪个岗位**、
**不告诉你缺了哪块硬技能**、**不告诉你简历哪句话在面试里会撑不住**。

这个工具的定位是**匹配与差距分析**：
读 JD → 读简历 → 逐条对齐 → 指出差距并给出可执行动作 → 预判面试会问什么。

它的输出长这样（真实产物，见 `experiments/results/demo/`）：

```
## 三、匹配分析

| 状态 | 技能 | 重要度 | 简历原句证据 | 判定说明 |
|------|------|--------|--------------|---------|
| ✅ 命中 | Agent | 必须 | 48 小时黑客松全程 AI 辅助开发（Codex 多智能体）… | 简历中有带动作/量化的项目级证据 |
| ❌ 缺失 | RAG | 必须 | （简历中无任何痕迹） | 简历中未出现该技能的任何痕迹 |

## 四、差距与优先级建议

1. **RAG**（必须，优先级 3.0）〔缺失〕
   - **具体动作**：在简历里补一条用 RAG 做过**可量化**产出（模型规模 / 数据量 / 精度提升 / 推理耗时），格式：「基于 RAG 实现 X，达到 Y」
   - **依据**：JD 要求「RAG」（必须），简历中缺失。JD 原文：了解大模型应用开发，熟悉 Prompt 设计、RAG 检索增强…
```

---

## 六步功能

| 步骤 | 做什么 | 实现方式 | 关键设计 |
|------|--------|----------|----------|
| 1 | **JD 技能要求抽取** | LLM（DeepSeek）+ **规则兜底** | 证据必须是 JD 原文的**逐字子句**；兜底条目标 `source=rule_fallback`，全程可追溯 |
| 2 | **简历解析** | 规则（.md/.txt/.docx） | .docx **真读 XML** 保留段落边界；每项技能都带原文证据 |
| 3 | **匹配分析** | 规则 + MCP 检索 | 命中/部分命中/缺失三档 + 简历原句证据；**假绿防线**（见下） |
| 4 | **差距与优先级建议** | 规则模板 | 按「重要度 × 缺失度」排序，每条给**具体动作**而非「建议学 X」 |
| 5 | **简历改写建议** | 规则 | 缺失项给结构建议；含糊表述给 before/after 示例 |
| 6 | **预测面试题** | 规则 | 按风险排序：**优先问简历上写得含糊/缺证据的地方** |

### ★ 假绿防线（本项目最重要的设计）

「简历里出现过这个词」≠「会这个技能」。

最典型的假绿：简历写「熟悉 Transformer」，JD 要求「深入理解 Transformer
并有大规模预训练经验」——只做词面匹配就会报「命中」，而候选人其实是
**做** Transformer 的（下游微调），不是**训**预训练模型的。面试当场穿帮。

所以匹配不只看词是否出现，还看：
1. 证据句是否含**动作动词**（用过/实现/开发/优化/训练…）；
2. 是否带**量化结果**（数字 + 指标）；
3. 出现在**项目/经历**还是只在「专业技能」清单里。

另外还有**歧义排除**（`jobfit/lexicon.py` 的 `AMBIGUITY_EXCLUSIONS`）：

- 「多轮超参**网格搜索**」里的「搜索」是 grid search，不是搜索算法；
- 「5 类工业纹理**量化对比**实验」里的「量化」是定量对比，不是模型量化。

这两类字面命中会直接造出假绿，所以要在匹配层就挡掉。

**实测：123 条「命中」结论中，假绿 0 次**（判据见[评测结果](#评测结果)）。

---

## 快速开始

### 环境准备

```bash
# Python 3.12（本项目在 3.12.9 上开发）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 核心功能零第三方依赖（只用标准库）
pip install -e .

# 可选增强：.docx 解析
pip install -e ".[docx]"
# 可选增强：MCP 检索通道（需要三个兄弟仓库，见下）
pip install -e ".[mcp,siblings]"
```

### 用起来

```bash
# 分析一份 JD + 一份简历，输出 Markdown 报告
python -m jobfit analyze \
  --jd data/jds/authored_nlp_classification.json \
  --resume data/resumes/简历202607-AI工程.md \
  --format md

# 输出 JSON（机器可读）
python -m jobfit analyze --jd <jd> --resume <resume> --format json --out report.json

# 纯规则模式（不调 LLM，不花钱）
python -m jobfit analyze --jd <jd> --resume <resume> --llm none

# HTTP 服务：/healthz + /analyze
python -m jobfit serve --port 8765
curl http://127.0.0.1:8765/healthz
curl -X POST http://127.0.0.1:8765/analyze \
  -H "Content-Type: application/json" \
  -d '{"jd":"data/jds/authored_eval_platform.json","resume":"data/resumes/简历202607-NLP算法.md","format":"md"}'

# 环境自检：兄弟仓库 / LLM / 依赖是否就绪
python -m jobfit doctor
```

### 产出

- **Markdown 报告**：人能读、能直接贴进求职流程（六步各一节 + 编排证据附录）
- **JSON**：机器可读，字段与 `jobfit/schemas.py` 一一对应
- **JSONL trace**：`traces/` 下，每次运行一个文件，每一步都可回放

---

## 评测结果

> 全部数字来自 `experiments/results/eval_summary.md` 与
> `experiments/results/eval_results.json`（由 `scripts/run_eval.py` 产出）。
> **本节只引用文件里的数字，不另造数字。**

复现：

```bash
python scripts/run_eval.py --llm deepseek --budget-usd 1.0 --out experiments/results
python scripts/run_demo.py --llm deepseek --budget-usd 0.5
```

### 指标 1：JD 技能要求抽取（P/R/F1）

数据集：12 份 JD（9 份公开真实 + 3 份自写变体），共 130 条人工标注。

| 口径 | P | R | F1 | 加权 P | 加权 R | 加权 F1 |
|---|---|---|---|---|---|---|
| 系统（LLM + 规则兜底） | 0.7829 | 0.9182 | **0.8452** | 0.7823 | 0.9238 | **0.8472** |
| 纯 LLM（关掉规则合并） | 0.8778 | 0.7182 | 0.7900 | 0.8830 | 0.7190 | 0.7926 |
| 关键词基线（无 LLM） | 0.7863 | 0.8364 | 0.8106 | 0.7807 | 0.8476 | 0.8128 |

- 系统 − 基线：micro F1 **+0.0346**，加权 F1 **+0.0344**
- 纯 LLM − 基线：micro F1 **−0.0206**

**这三行怎么读（这是本评测最值得看的部分）：**

- **纯 LLM 这一档低于基线** —— 说明 LLM 单独用会**漏词**
  （recall 0.7182 vs 基线 0.8364）：它只挑「显眼」的技能，
  漏掉藏在长句里的那些（分布式系统设计、数据库设计…）。
- **系统档反超基线** —— 靠的是规则兜底把漏的补回来：
  规则 recall 0.8364，LLM precision 0.8778，**两者互补**。
- 所以结论不是「LLM 比规则强」，而是：**LLM 精确、规则全面，合并后最好**。
  这也是本项目坚持「LLM + 规则兜底、兜底全程标记」的真实原因。

### 指标 2：匹配环节的证据定位正确率

判据（两条都要过才算对）：
① 证据句是简历原文的子串；② 证据句里确实出现该技能。

- 有证据的结论：**151** 条
- 两项都对：**151** 条 → **正确率 100%**

### 指标 3：端到端「推荐哪版简历」准确率

- **9 / 12** 份 JD 推荐正确（准确率 0.75）
- 期望版本按**岗位族**人工标注（4 份简历 ↔ 4 个岗位族），系统推荐与该标签相互独立

### 指标 4：假绿次数

- 判据：系统判「命中」，但**人工标注真值**判该简历不具备该技能
- 期间系统共给出「命中」结论 **123** 条
- **假绿 0 次**

> 真值表是**人工标注**的（`data/annotations/resume_skill_truth.json`），
> 与系统匹配逻辑完全独立——否则系统说命中、系统也说命中，假绿数恒为 0，指标就废了。

### 指标 5：兜底占比与 LLM 费用

- 兜底条数 **39** / 总条数 **129** → 兜底占比 **30.2%**
- LLM 调用 12 次，token 12440 输入 + 6412 输出，估算费用 **$0.0057**

### 端到端 Demo

真实 JD（腾讯「光子 AI-AI应用开发工程师」，来源链接与抓取日期见
`data/jds/2008789833664847872.json`）+ 4 份真实脱敏简历 → 4 份完整六步报告：

| 简历 | 命中 | 部分命中 | 缺失 | 差距建议 | 改写建议 | 面试题 |
|------|------|----------|------|----------|----------|--------|
| AI工程 | 2 | 0 | 10 | 10 | 13 | 8 |
| Java开发 | 1 | 0 | 11 | 11 | 16 | 8 |
| NLP算法 | 1 | 1 | 10 | 11 | 17 | 8 |
| 评测后训练-有联 | 0 | 0 | 12 | 12 | 15 | 8 |

**推荐版本：AI工程 版**（命中 2 条、缺失 10 条，为四份里最优）。

报告在 `experiments/results/demo/`：
`demo_2008789833664847872_*.md`（4 份）+ `_summary.md`（横向对比）+ `demo_run.json`（机器可读）。

---

## 与三个地基仓库的组合

这个项目**不是**三个孤岛。它真的把三个已有仓库组合起来用：

| 地基仓库 | 提供什么 | 本项目怎么用 |
|----------|----------|--------------|
| **hello-agent** | 韧性 + JSONL trace | `agent.trace.Tracer`（每步落 JSONL，可回放）、`agent.resilience.ToolExecutor`（超时/重试/降级/熔断，保护 MCP 工具调用） |
| **rag-mcp-server** | 检索引擎 | `rag.chunking` / `rag.embedding` / `rag.retrieval` / `rag.prompt_guard` —— 简历分块、向量检索、JD 条款检索的**实现**直接复用，不重写 |
| **rag-eval-platform** | 评测与报告口径 | 复用其「降级必须被看见」「不编造数字」的评测纪律与报告结构 |

**组合是真的，不是说说**：

- 检索能力经由 **MCP stdio 协议**调用（`jobfit/mcp_client.py` ↔ `jobfit/mcp_server.py`），
  不是进程内直接调函数——走一遍协议才能验证这条链路真的通；
- `jobfit/mcp_server.py` 里 `from rag.chunking import FixedSizeChunker` 这样的 import
  是**真的**指向兄弟仓库的源码；
- 本仓库的 git 历史里**没有**任何兄弟仓库的代码副本。

**没有直接复用 hello-agent 的 ReAct 图**，这是有意的：
本项目的六步是**固定流程**（抽取 → 解析 → 匹配 → 建议 → 面试题），
而 hello-agent 的 `build_agent_graph` 是**LLM 自主决定下一步**的 ReAct 循环。
硬套会让「同一份输入跑出不同结果」，评测就没法做了。
所以复用的是它的 trace + resilience 两层（真正的复用），
编排则用一个**确定性的多步流水线**实现。

---

## 目录约定

三个兄弟仓库以**本地路径**放在应用仓库**旁边**：

```
<parent>/
  job-fit-assistant/     ← 本仓库
  hello-agent/           ← Agent 框架（trace / 韧性）
  rag-mcp-server/        ← MCP 工具服务（检索引擎）
  rag-eval-platform/     ← 评测平台
```

默认 `<parent>` = 本仓库的上一级目录。也可以用环境变量覆盖：

```bash
export JOBFIT_SIBLINGS_ROOT=/path/to/parent
```

**CI 里怎么保证**：`.github/workflows/ci.yml` 用三个 `actions/checkout` 把四个仓库


> **Gitee 侧也接了自动门禁**：`.workflow/MasterPipeline.yml`（Gitee Go），**同一批命令、同一套口径**（含同样的排除清单），每次 push 到 master 自动构建；构建记录在仓库页「流水线」里（**需登录可见**，故不挂徽章）。
checkout 到**同一父目录**（`path:` 参数 + `../` 相对路径），
然后 `pip install -e ".[dev,mcp,siblings]"` 装上依赖再跑 pytest。

**找不到兄弟仓库时**：不报错退出，而是**降级**——
规则路径的抽取、匹配、建议、面试题全部照常工作，只是少了 MCP 检索通道
（此时证据定位走本地词表匹配，精度会下降，报告里会标明「检索通道降级」）。

```bash
python -m jobfit doctor   # 一眼看清楚哪些就绪、哪些降级
```

---

## 隐私与脱敏

**入库的简历全部脱敏过。** 原始 `.docx` / `.pdf` **一律不入库**，
只放脱敏后的 `.md`。

脱敏口径（`scripts/anonymize_resume.py`）：

| 类型 | 处理 | 例子（虚构值） |
|------|------|------|
| 姓名 | 替换为「候选人」 | 张三 → 候选人 |
| 手机号 | 整段移除 | 138xxxxxxxx → （删除） |
| 邮箱 | 整段移除 | someone@example.com → （删除） |
| 身份证号 | 整段移除 | 18 位 → （删除） |
| 学号 | 整段移除 | （删除） |
| 期望薪资 | 整行移除 | 「期望薪资：XX-XXK」→ 删除该行 |
| 目标公司名 | 移除（保留岗位方向） | 「求职：某公司 后端工程师」→「求职： 后端工程师」 |
| 学校 / 项目经历 / 技术栈 | **保留** | 见下 |

**为什么保留学校与项目**：这些是**岗位匹配必需**的信息，抹掉简历就没有评测价值了。
姓名 + 联系方式 + 薪资期望才是「足以直接定位到具体的人」的部分，脱敏的目标是后者。

**为什么必须脱敏**：本仓库要开源。简历里的姓名、手机号、邮箱、薪资期望
组合起来足以直接定位到真人；提交上去等于把个人信息公开。
而且对招聘方来说，看到真实个人信息的开源仓库，可信度会直接归零。

私密文件（`求职作战板.md`、`personal-background.md`）**一律不入库**。

自证（一条命令；CI 里跑的是同一份脚本）：

```bash
bash scripts/privacy_check.sh
```

脚本查四项，全部 0 命中才算通过：

| 检查 | 查什么 | 为什么这么查 |
|------|--------|--------------|
| 敏感串（data/） | 手机号形态、邮箱形态、薪资行 | 真实个人信息只可能出现在数据里；手机号模式带**前后非数字边界**，否则 19 位的腾讯 postId 会被误判；邮箱排除 RFC 保留的 `example.*` |
| 代码不硬编码个人信息 | 手机号、邮箱 | 脱敏脚本里的「薪资」是**脱敏规则的正则定义**（不是数据），允许；手机号/邮箱不允许 |
| 绝对路径 | `C:\Users\...`、`/home/xxx`、`/Users/xxx` | 提交物不能带本机路径（泄露目录结构，也不可移植） |
| 私密文件未入库 | `求职作战板`、`personal-background`、`.docx`、`.pdf` | 这两类东西绝不能进版本库 |

---

## 评测数据集

### 12 份 JD

`data/jds/*.json`，每份带**人工标注**的技能要求 + 重要度 + 原文证据。

| 类型 | 份数 | 来源 |
|------|------|------|
| 公开真实 JD | **9** | [腾讯招聘官网](https://careers.tencent.com/jobdesc.html)公开接口，逐份附**来源链接**与**抓取日期**（2026-10-21） |
| 自写变体 | **3** | 基于真实 JD 的结构与内容改写，`source_type: "authored_variant"` 明确标注，**不冒充真实 JD** |

九份真实 JD 的岗位方向（覆盖四个岗位族，让「推荐哪版简历」这道题有区分度）：

| JD | 岗位 | 来源链接 |
|----|------|----------|
| 2008789833664847872 | 光子 AI-AI应用开发工程师 | http://careers.tencent.com/jobdesc.html?postId=2008789833664847872 |
| 2108081571104997376 | AI 应用开发高级工程师（深圳/北京） | http://careers.tencent.com/jobdesc.html?postId=2108081571104997376 |
| 2098594908700061696 | 云mall-java后端开发工程师-电商 | http://careers.tencent.com/jobdesc.html?postId=2098594908700061696 |
| 2089960263330742272 | 营销与交易产品部-云mall高级Java后端开发工程师 | http://careers.tencent.com/jobdesc.html?postId=2089960263330742272 |
| 2092134938500579328 | 搜狗输入法-NLP算法工程师 | http://careers.tencent.com/jobdesc.html?postId=2092134938500579328 |
| 2082710394723545088 | 微信小程序-大模型算法工程师-Coding方向 | http://careers.tencent.com/jobdesc.html?postId=2082710394723545088 |
| 2094265700456771584 | MaaS模型评测高级工程师 | http://careers.tencent.com/jobdesc.html?postId=2094265700456771584 |
| 1915233782491881472 | 混元大模型评测算法研究员（北京） | http://careers.tencent.com/jobdesc.html?postId=1915233782491881472 |
| 2095418906612248576 | 微信小程序-大模型算法工程师-智能体方向 | http://careers.tencent.com/jobdesc.html?postId=2095418906612248576 |

**抓取留痕**：`scripts/fetch_jds.py` 抓取 → `data/jds/raw/tencent_raw.json` 存原始结果
（任何人重跑都能核对是不是同一份文本）。

### 标注口径

写在 `data/jds/*.json` 的 `annotations.rubric` 里，也抄在这里：

1. **什么算一条要求**：JD 里明确写出的「技能/技术/工具/方法/能力」，
   且是该岗位要招人具备的东西。
   **排除**：学历与年限要求、软素质（沟通/责任心/学习能力）、
   纯岗位职责里的业务描述、薪酬福利。
2. **重要度判定**：
   - 「必须」= 原文出现 精通/熟练掌握/扎实/深入理解/具备…经验/至少… 等硬性表述，
     或写在任职要求里的无条件句；
   - 「加分」= 原文出现 优先/加分/更佳/有兴趣/…者优先；
   - 一句里两者都有（「熟悉 X，有 Y 经验者优先」）→ 判「必须」（硬性条件在前）。
3. **技能名归一化**：用 `jobfit.normalize` 的口径（词表映射 + 复合短语拆分），
   这样 LLM 抽取、规则抽取、人工标注三方能对齐到同一集合上。
4. **原文证据**：必须是 JD 原文的**逐字片段**；
   `scripts/build_dataset.py` 会校验，定位不到**直接报错**——标注与原文脱节必须炸出来。

### 4 份脱敏简历

`data/resumes/*.md`（从真实 `.docx` 转 `.md`）：
AI工程 / Java开发 / NLP算法 / 评测后训练 —— 对应四个岗位族。

### 人工标注真值表

`data/annotations/resume_skill_truth.json`：「技能 × 简历」的**具备/不具备**判定
（4 × 62 = 248 个判定）。

这份表是**假绿指标**的唯一依据——它由人独立标注，与系统的匹配逻辑无关。
标注口径：

- **具备**：简历里有该技能的「技术栈行」，或「带动作动词/量化结果的项目句」；
- **不具备**：完全没有该技能痕迹，或只出现在「了解/熟悉/学过」这类无产出表述里。

---

## LLM 与费用

| 后端 | 何时启用 | 说明 |
|------|----------|------|
| `deepseek` | 有 API key 时（默认） | OpenAI 兼容端点，**只用标准库 urllib**，不引 SDK |
| `local` | 没有 key 时 | 本机离线 Qwen2.5-1.5B-Instruct（HF 缓存里已有，不下载） |
| `none` | `--llm none` | 纯规则，不调 LLM，不花钱 |

**Key 来源**：`DEEPSEEK_API_KEY` → `OPENAI_API_KEY` → `ANTHROPIC_AUTH_TOKEN`（按此顺序取）。

**预算上限**：`--budget-usd`（默认 0.5）。检查在**调用前**做（预估输入 token + 上限输出 token），
超了直接抛 `BudgetExceeded`，**不做「先花钱再发现超了」**。

**费用记账**：`experiments/results/eval_results.json` 的 `metric_5_fallback_and_cost.llm_usage`。
价目表来自 [DeepSeek 官方文档](https://api-docs.deepseek.com/quick_start/pricing)（抓取日 2026-10-21），
**统一按非峰值档记账**（保守方向：非峰值价 ≤ 峰值价，报出来的费用不会偏高）。
价目表未命中的模型**只报 token 不报钱**——不编造成本。

**本次评测实测**：12 次调用，token 12440 + 6412，**$0.0057**。

**离线优先**：所有检索与匹配逻辑在 `--llm none` 下照常工作，LLM 只用来提升抽取精度。
本机的 BGE embedding 模型与 Qwen 都是离线缓存的，不下载新模型。

---

## 诚实边界

写在前面，避免把量级当结论：

1. **n = 12 份 JD × 4 份简历**。样本量小，所有比例**只应作为量级参考**，
   不要读成「准确率 90%」这种断言。
2. **人工标注是单人标注**，没有第二标注者做交叉复核。
   重要度的边界案例（「熟悉 X，有 Y 经验者优先」）按上文规则统一判定。
3. **端到端推荐的期望版本是人工标签**（按岗位族），人写标签有主观性。
   试过用真值表客观推导，但退化了（评测后训练版覆盖面最广，
   几乎每份 JD 都推导出它，测不出岗位匹配能力）——那次失败写在代码注释里。
4. **关键词基线用同一份技能词表**，它无法区分「必须/加分」，
   因此在重要度加权口径上**必然**吃亏——这是基线设计使然，不是实现缺陷。
5. **词表是人工维护的领域词表**，覆盖本评测集涉及的技术方向，
   但它天然不覆盖所有技术名词（词表外技能走「保留原文 + LLM 判定类别」路径）。
6. **真值表的「具备」门槛带主观性**（本表按「有技术栈行即算具备」处理，
   比「必须要有项目级产出」宽松）——这让假绿指标偏保守。
7. **LLM 成本是估算**，按公开价目表的非峰值档算，实际账单可能因时段不同而变化。
8. **真实求职里「哪版简历更好」还取决于公司、薪资、地域、投递时机**，
   这些都不在评测范围内——本工具只回答「技术上更匹配哪版」。

**不编造数字**：报告里的每个数字都来自 `experiments/results/` 下的文件。
跑不出来就说跑不出来，不写「大概」「应该」。

---

## 开发

```bash
pip install -e ".[dev]"
pytest -q          # 61 个单测，不联网、不调 API、不加载模型
```

### 目录

```
jobfit/
  lexicon.py      技能词表（105 条）+ 别名反查 + 歧义排除
  normalize.py    技能名归一化（抽取器与评测器共用，保证能对齐）
  filters.py      非技能项过滤
  schemas.py      六步流程的结构化数据结构
  llm.py          LLM 客户端（deepseek / local / none）+ 预算上限 + 费用记账
  rules.py        规则兜底 + 关键词基线
  extractor.py    步骤 1：JD 技能要求抽取
  resume.py       步骤 2：简历解析（.md/.txt/.docx）
  matching.py     步骤 3：匹配分析 + 假绿防线
  advice.py       步骤 4：差距与优先级建议
  rewrite.py      步骤 5：简历改写建议
  interview.py    步骤 6：预测面试题
  mcp_server.py   MCP 服务：简历分块/检索、JD 条款检索（复用 rag-mcp-server 的 rag 包）
  mcp_client.py   MCP 客户端（真走 stdio 协议）
  siblings.py     三个兄弟仓库的本地路径集成
  pipeline.py     六步编排 + trace + 韧性
  report.py       Markdown / JSON 报告
  server.py       HTTP 服务（stdlib http.server，零新依赖）
  metrics.py      五个评测指标的计算与口径
  cli.py          CLI 入口
scripts/
  fetch_jds.py            抓公开真实 JD
  build_dataset.py        建评测集（含标注证据校验）
  anonymize_resume.py    简历脱敏
  annotate_resume_truth.py  生成人工标注真值表
  run_eval.py             跑完整评测
  run_demo.py             端到端 demo
data/
  jds/          12 份 JD（含人工标注）+ raw/ 抓取留痕
  resumes/      4 份脱敏简历
  annotations/  人工标注真值表
experiments/results/   评测数字 + demo 报告
```

### 重新生成数据与评测

```bash
python scripts/fetch_jds.py --out data/jds/raw --limit 10
python scripts/build_dataset.py
python scripts/anonymize_resume.py --in-dir <简历目录> --out-dir data/resumes
python scripts/annotate_resume_truth.py
python scripts/run_eval.py --llm deepseek --budget-usd 1.0 --out experiments/results
python scripts/run_demo.py --llm deepseek --budget-usd 0.5
```

---

## License

MIT
