# -*- coding: utf-8 -*-
"""解析你填好的 REVIEW_SHEET.md → 一致性报告 + 修正后的真值表。

输出：
  data/annotations/second_pass_review.json   逐条判定（同意/改判/删除/漏标）+ 修正后的 gold
  experiments/results/second_pass_agreement.md 给人看的一致性报告
"""
import json
import re
from collections import Counter
from pathlib import Path

P = Path(r"D:\programmer\LLM\projects\job-fit-assistant")
SHEET = P / "data" / "jds" / "REVIEW_SHEET.md"
OUT_JSON = P / "data" / "annotations" / "second_pass_review.json"
OUT_MD = P / "experiments" / "results" / "second_pass_agreement.md"

text = SHEET.read_text(encoding="utf-8")

# 按 "## N. 标题  `id`" 切段
sections = re.split(r"\n## \d+\.\s", text)[1:]
print(f"[解析] {len(sections)} 个 JD 段")

per_jd = []
for sec in sections:
    head = sec.splitlines()[0]
    m_id = re.search(r"`([0-9a-z_]+)`", head)
    m_title = re.sub(r"\s*`[0-9a-z_]+`.*$", "", head).strip()
    jd_id = m_id.group(1) if m_id else "?"
    title = m_title

    # 原文块
    raw_block = re.search(r"```text\n(.*?)\n```", sec, re.S)
    raw_text = raw_block.group(1) if raw_block else ""

    # 标注块（第二个 ```text 块）
    blocks = re.findall(r"```text\n(.*?)\n```", sec, re.S)
    ann_block = blocks[1] if len(blocks) > 1 else ""

    items = []
    cur = None
    for line in ann_block.splitlines():
        m_item = re.match(r"\[(\d+)\]\s*技能=(.*?)\s+类别=(.*?)\s+重要性=(.*?)\s*$", line)
        m_add = re.match(r"\[补\]\s*技能=(.*?)(?:\s+证据=(.*))?$", line)
        m_ev = re.match(r"\s*证据=(.*)$", line)
        m_verdict = re.match(r"\s*复核=(.*)$", line)
        if m_item:
            cur = {"n": int(m_item.group(1)), "skill": m_item.group(2), "category": m_item.group(3),
                   "importance": m_item.group(4), "evidence": None, "verdict": None}
            items.append(cur)
        elif m_add and not m_item:
            items.append({"n": None, "skill": m_add.group(1).strip(), "category": None,
                          "importance": None, "evidence": (m_add.group(2) or "").strip(),
                          "verdict": "补", "added": True})
        elif m_ev and cur is not None:
            cur["evidence"] = m_ev.group(1).strip()
        elif m_verdict and cur is not None:
            cur["verdict"] = m_verdict.group(1).strip() or None
    per_jd.append({"id": jd_id, "title": title, "raw_text": raw_text, "items": items})

# 统计
total_auto = sum(len([i for i in jd["items"] if not i.get("added")]) for jd in per_jd)
total_added = sum(len([i for i in jd["items"] if i.get("added")]) for jd in per_jd)
verdicts = Counter()
unchanged = changed = deleted = unfilled = 0
for jd in per_jd:
    for it in jd["items"]:
        if it.get("added"):
            continue
        v = it["verdict"]
        if v is None:
            unfilled += 1
            verdicts["未填"] += 1
        elif v.startswith("同意"):
            unchanged += 1
            verdicts["同意"] += 1
        elif v.startswith("删除"):
            deleted += 1
            verdicts["删除"] += 1
        else:
            changed += 1
            verdicts["改判"] += 1

filled = unchanged + changed + deleted
agree_pct = f"{unchanged / filled * 100:.1f}%" if filled else "—"
print(f"[统计] 自动标注 {total_auto} 条 | 你填了 {filled} 条（未填 {unfilled}）| 同意 {unchanged} / 改判 {changed} / 删除 {deleted} | 你补了 {total_added} 条")

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "source_sheet": "data/jds/REVIEW_SHEET.md",
    "jd_count": len(per_jd),
    "auto_annotations": total_auto,
    "human_filled": filled,
    "human_unfilled": unfilled,
    "verdicts": dict(verdicts),
    "human_added": total_added,
    "per_jd": [{k: v for k, v in jd.items() if k != "raw_text"} for jd in per_jd],
}, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
print(f"[写出] {OUT_JSON.relative_to(P)}")

lines = [
    "# 第二标注者复核 · 一致性报告", "",
    f"- 来源复核表：`data/jds/REVIEW_SHEET.md`（{len(per_jd)} 份 JD）",
    f"- 自动标注 **{total_auto}** 条；你填了 **{filled}** 条（未填 {unfilled}）",
    f"- 结论：**同意 {unchanged}**（{agree_pct}）｜**改判 {changed}**｜**删除 {deleted}**｜**你补了 {total_added}** 条",
    "",
]
if filled:
    lines += ["## 逐条判定", ""]
    for jd in per_jd:
        bad = [i for i in jd["items"] if not i.get("added") and (i["verdict"] is None or not i["verdict"].startswith("同意"))]
        add = [i for i in jd["items"] if i.get("added")]
        if not bad and not add:
            continue
        lines.append(f"### {jd['title']}（`{jd['id']}`）")
        lines.append("")
        for i in bad:
            mark = "未填" if i["verdict"] is None else i["verdict"]
            lines.append(f"- [{i['n']}] {i['skill']}（{i['category']} / {i['importance']}）→ **{mark}**　证据「{i['evidence']}」")
        for i in add:
            lines.append(f"- ➕ 补：{i['skill']}　证据「{i['evidence']}」")
        lines.append("")
else:
    lines += ["> 复核表还没填（或填了但没保存）—— 填完再跑这个脚本。", ""]

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.write_text("\n".join(lines), encoding="utf-8", newline="\n")
print(f"[写出] {OUT_MD.relative_to(P)}")
