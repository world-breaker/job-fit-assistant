# -*- coding: utf-8 -*-
"""给 job-fit-assistant 的自检加一条「标注条数从数据里数」的守卫，并修 README 里虚高的 130。

起因（2026-10-21 实测）：README 写「共 130 条人工标注」，而从 `data/jds/*.json` 的
`annotations.requirements` 实际数出来是 **119 条**（必须 108 / 加分 11）——
**数字必须从数据里当场数，不能凭印象写**。所以把它做成自检里的一条 claim：
以后标注集变了、README 没跟着改，自检会直接报错。
"""
import json
from pathlib import Path

P = Path(r"D:\programmer\LLM\projects\job-fit-assistant")
JD = P / "data" / "jds"
CHECK = P / "scripts" / "check_doc_numbers.py"
README = P / "README.md"

# ---------- 1) 数出真实条数 ----------
files = sorted(f for f in JD.glob("*.json") if f.name != "index.json")
total = must = 0
for f in files:
    d = json.loads(f.read_text(encoding="utf-8"))
    for r in d["annotations"]["requirements"]:
        total += 1
        if r["importance"] == "必须":
            must += 1
plus = total - must
print(f"[真值] {len(files)} 份 JD，技能标注 {total} 条（必须 {must} / 加分 {plus}）")

# ---------- 2) 修 README ----------
raw = README.read_bytes()
crlf = raw.count(b"\r\n") > 0
text = raw.decode("utf-8").replace("\r\n", "\n")
OLD = "共 130 条人工标注。"
NEW = f"共 **{total}** 条人工标注（必须 {must} / 加分 {plus}）。"
if OLD in text:
    text = text.replace(OLD, NEW, 1)
    README.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print(f"[README] 已把「130 条」改成「{total} 条（必须 {must} / 加分 {plus}）」")
elif f"共 **{total}** 条人工标注" in text:
    print("[README] 已是正确数字")
else:
    print("★ README 未命中目标表述（需人工看）")

# ---------- 3) 自检加一条 claim ----------
raw = CHECK.read_bytes()
crlf = raw.count(b"\r\n") > 0
text = raw.decode("utf-8").replace("\r\n", "\n")
if "标注条数" in text:
    print("[自检] 该 claim 已存在")
else:
    anchor = "def expected_numbers() -> List[Tuple[str, str]]:"
    inject = '''def annotated_skill_count() -> Tuple[str, str]:
    """从 data/jds/*.json 的 annotations.requirements **当场数**出技能标注条数。

    ★ 为什么必须当场数：2026-10-21 实测，README 写「共 130 条人工标注」，
    而真实条数是 119（必须 108 / 加分 11）—— 数字得从数据里来，不能凭印象写。
    """
    jd_dir = ROOT / "data" / "jds"
    total = must = 0
    for f in sorted(jd_dir.glob("*.json")):
        if f.name == "index.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for r in d.get("annotations", {}).get("requirements", []):
            total += 1
            if r.get("importance") == "必须":
                must += 1
    return str(total), f"必须 {must} / 加分 {total - must}"


'''
    if anchor not in text:
        raise SystemExit("❌ 自检脚本里没找到 expected_numbers 锚点")
    text = text.replace(anchor, inject + anchor, 1)
    # 在 expected_numbers 的 return 列表里加一项
    ret_anchor = '    return [\n'
    if ret_anchor not in text:
        raise SystemExit("❌ 找不到 return 列表锚点")
    add = ('        ("技能标注条数", annotated_skill_count()[0]),\n'
           '        ("技能标注构成", annotated_skill_count()[1]),\n')
    text = text.replace(ret_anchor, ret_anchor + add, 1)
    CHECK.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print("[自检] 已加两条 claim：技能标注条数 / 技能标注构成")
