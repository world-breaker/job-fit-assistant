# -*- coding: utf-8 -*-
"""生成「单文件人工复核表」：12 份 JD 各自的原文 + 自动标注 + 待你填的复核行。

设计要点（为了让人能真的填）：
  - **一个文件**（不用开 12 个 json），每个 JD 一段；
  - 标注区放在**代码块**里（编辑器里不会被 markdown 表格搅乱），行格式极简、可直接改；
  - 每条已预填：技能 / 类别 / 重要性 / 原文证据；只需在「复核=」后写结论；
  - 原文整段放在上面的代码块里，方便对照（标注必须与原文逐字对得上）。
"""
import json
from pathlib import Path

P = Path(r"D:\programmer\LLM\projects\job-fit-assistant")
JD = P / "data" / "jds"
OUT = JD / "REVIEW_SHEET.md"

files = sorted(f for f in JD.glob("*.json") if f.name != "index.json")

HEAD = """# JD 技能要求 · 人工复核表（第二标注者）

> **怎么用**：从上往下逐个 JD 看。① 先看「JD 原文」；② 看下面每条**已自动标注**的要求；
> ③ 在每条后面的 `复核=` 处写你的结论（三种）：
> - `同意` —— 这条要求、技能名、重要性都判对了
> - `改判:重要性=加分` —— 技能名对但重要性判错（可写 `改判:技能=XXX` 改技能名，或 `改判:重要性=加分,技能=XXX`）
> - `删除` —— 这条**不算**一条要求（学历年限 / 软素质 / 纯业务描述 / 薪酬福利）
> ④ 发现**漏掉**的要求，在该 JD 段落末尾加一行：`[补] 技能=XXX  证据=（原文逐字片段）`
>
> **判据**（`data/jds/ANNOTATION.md` 是完整版，这里是要点）：
> - **算**一条要求：JD 明确写出的「技能/技术/工具/方法/能力」，且是这个岗位要招人具备的东西。
>   复合短语要**拆开**（「熟悉 Spring Boot、Spring Cloud、MySQL」→ 3 条）。
> - **不算**：学历与年限要求、软素质（沟通/责任心/抗压）、纯业务描述、薪酬福利、会议与期刊名。
> - **重要性**：出现「精通/熟练掌握/扎实/深入理解/具备…经验/至少…」等硬性表述 → **必须**；
>   出现「优先/加分/更佳/有兴趣」 → **加分**；「熟悉 X，有 Y 经验者优先」→ 统一判 **必须**。
> - **技能名**用 `jobfit.normalize` 的规范名（词表别名映射 + 复合短语拆分）；词表里没有的写原文。
> - **证据**必须是 JD 原文的**逐字片段**（短片段即可，不必整句）。
>
> 填完存成同名文件（或直接改这个文件），跟我说一声，我跑一致性统计：
> 自动标注 vs 你的复核 —— 逐条一致率、你的改判清单、漏标清单，以及**以你的标注为真值**重算抽取指标。

"""

parts = [HEAD]
total = 0
for idx, f in enumerate(files, 1):
    d = json.loads(f.read_text(encoding="utf-8"))
    reqs = d["annotations"]["requirements"]
    total += len(reqs)
    must = sum(1 for r in reqs if r["importance"] == "必须")
    src = d.get("source_url", "")
    src_note = "公开真实 JD" if d.get("source_type") == "public_real" else "自写变体（不算真实 JD）"
    parts.append(f"\n---\n\n## {idx}. {d.get('title','')}  `{d.get('id','')}`\n")
    parts.append(f"- 来源：{src}（{src_note}，抓取日 {d.get('fetched_at','')}）\n")
    parts.append(f"- 期望简历版本：`{d.get('expected_resume','')}`\n")
    parts.append(f"- 自动标注 **{len(reqs)}** 条（必须 {must} / 加分 {len(reqs)-must}）\n")
    parts.append("\n**JD 原文**\n\n```text\n" + d.get("raw_text", "").rstrip() + "\n```\n")
    parts.append("\n**待复核的标注**（在每条的 `复核=` 后写结论）\n\n```text\n")
    for i, r in enumerate(reqs, 1):
        parts.append(f"[{i}] 技能={r['skill']}  类别={r['category']}  重要性={r['importance']}\n")
        parts.append(f"     证据={r.get('anchor') or r.get('evidence','')}\n")
        parts.append(f"     复核=\n")
    parts.append("```\n")

parts.append(f"\n---\n\n合计：{len(files)} 份 JD、{total} 条自动标注。\n")
OUT.write_text("".join(parts), encoding="utf-8", newline="\n")
print(f"✅ 复核表已生成：{OUT}（{OUT.stat().st_size} 字符，{len(files)} 份 JD，{total} 条待复核）")
