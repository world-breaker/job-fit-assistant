# -*- coding: utf-8 -*-
"""生成「抽样复核表」：每份 JD 抽 2 条（类别尽量分散）→ 24 条，用来估一致率。

★ 为什么给抽样版：全量 119 条要 1~2 小时；抽样 24 条只要 10~15 分钟，
  而「一致率」这个数字要回答的问题是「标注口径稳不稳」—— 24 条足够给出量级判断。
"""
import json
import re
from pathlib import Path

P = Path(r"D:\programmer\LLM\projects\job-fit-assistant")
JD = P / "data" / "jds"
OUT = JD / "REVIEW_SHEET_SAMPLE.md"
PER_JD = 2

files = sorted(f for f in JD.glob("*.json") if f.name != "index.json")

HEAD = """# JD 技能要求 · **抽样复核表**（第二标注者，24 条版）

> **这是抽样版**：12 份 JD，每份抽 **2 条**（尽量覆盖不同类别），共 **24 条**。
> 目标是估出「标注口径稳不稳」的一致率 —— 10~15 分钟能标完，结论够用。
> 想全量标（119 条）就用同目录的 `REVIEW_SHEET.md`。
>
> **怎么填**：在每条的 `复核=` 后写 `同意` / `改判:重要性=加分` / `删除`；
> 发现漏标就在该 JD 段落末尾加 `[补] 技能=XXX  证据=（原文逐字片段）`。
> 判据摘要见 `REVIEW_SHEET.md` 开头，或完整版 `data/jds/ANNOTATION.md`。
>
> 标完跟我说一声，我跑 `scripts/apply_review_sheet.py` 出逐条一致率与改判清单。

"""

parts = [HEAD]
picked_total = 0
for idx, f in enumerate(files, 1):
    d = json.loads(f.read_text(encoding="utf-8"))
    reqs = d["annotations"]["requirements"]
    # 抽 2 条：优先不同类别、且尽量包含「加分」条目（重要性边界最容易判错）
    chosen, used_cats = [], set()
    for pref in ("加分", None):
        for r in reqs:
            if r in chosen:
                continue
            if pref and r["importance"] != pref:
                continue
            if r["category"] in used_cats and len(chosen) < PER_JD:
                continue
            chosen.append(r)
            used_cats.add(r["category"])
            if len(chosen) >= PER_JD:
                break
        if len(chosen) >= PER_JD:
            break
    picked_total += len(chosen)

    src = d.get("source_url", "")
    src_note = "公开真实 JD" if d.get("source_type") == "public_real" else "自写变体（不算真实 JD）"
    parts.append(f"\n---\n\n## {idx}. {d.get('title','')}  `{d.get('id','')}`\n")
    parts.append(f"- 来源：{src}（{src_note}，抓取日 {d.get('fetched_at','')}）\n")
    parts.append(f"- 期望简历版本：`{d.get('expected_resume','')}`　｜　本份共 {len(reqs)} 条，**本次抽 {len(chosen)} 条**\n")
    parts.append("\n**JD 原文**\n\n```text\n" + d.get("raw_text", "").rstrip() + "\n```\n")
    parts.append("\n**待复核（抽到的条目）**\n\n```text\n")
    for i, r in enumerate(chosen, 1):
        parts.append(f"[{i}] 技能={r['skill']}  类别={r['category']}  重要性={r['importance']}\n")
        parts.append(f"     证据={r.get('anchor') or r.get('evidence','')}\n")
        parts.append("     复核=\n")
    parts.append("```\n")

parts.append(f"\n---\n\n合计：抽样 **{picked_total}** 条（全量是 119 条）。\n")
OUT.write_text("".join(parts), encoding="utf-8", newline="\n")
print(f"✅ 抽样复核表已生成：{OUT.name}（{OUT.stat().st_size} 字符，{picked_total} 条待标）")
