"""核对 README 里引用的数字与 experiments/results/ 下的实际文件是否一致。

★ 为什么要有这个脚本：README 的数字是「引用」结果文件的。
如果结果文件更新了而 README 没改，README 就在说谎——而且没人会发现。
这个脚本把两者对上，对不上就报错。

用法：
    python scripts/check_doc_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results" / "eval_results.json"
SUMMARY = ROOT / "experiments" / "results" / "eval_summary.md"
README = ROOT / "README.md"

#: README 里必须出现的关键数字（键 -> 期望出现在 README 里的字符串）
#:
#: ★ 这些值**不是抄来的**，而是从 eval_results.json 里算出来后写进 README 的；
#:   本脚本反过来验证「README 里的值 == 结果文件里的值」。
def expected_numbers() -> List[Tuple[str, str]]:
    if not RESULTS.exists():
        return []
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    m1 = data["metric_1_extraction"]
    m2 = data["metric_2_evidence_localization"]
    m3 = data["metric_3_end_to_end_resume_recommendation"]
    m4 = data["metric_4_false_green"]
    m5 = data["metric_5_fallback_and_cost"]

    usage = m5.get("llm_usage") or {}
    cost = f"{usage.get('cost_usd', 0.0):.4f}"

    return [
        ("系统 micro F1", f"{m1['system_llm_plus_fallback']['micro']['f1']:.4f}"),
        ("纯LLM micro F1", f"{m1['llm_only']['micro']['f1']:.4f}"),
        ("基线 micro F1", f"{m1['keyword_baseline']['micro']['f1']:.4f}"),
        ("系统 加权F1", f"{m1['system_llm_plus_fallback']['micro']['weighted_f1']:.4f}"),
        ("基线 加权F1", f"{m1['keyword_baseline']['micro']['weighted_f1']:.4f}"),
        ("F1 差值", f"{m1['delta_micro_f1']:+.4f}"),
        ("证据总数", str(m2["total_conclusions_with_evidence"])),
        ("证据正确数", str(m2["both_correct"])),
        ("端到端正确", f"{m3['correct']} / {m3['jd_count']}"),
        ("假绿次数", str(m4["count"])),
        ("命中结论总数", str(m4["total_hit_conclusions"])),
        ("兜底条数", str(m5["fallback_requirements"])),
        ("总条数", str(m5["total_requirements"])),
        ("兜底占比", f"{m5['fallback_ratio'] * 100:.1f}%"),
        ("LLM 费用", f"${cost}"),
        ("LLM token 输入", str(usage.get("prompt_tokens", 0))),
        ("LLM token 输出", str(usage.get("completion_tokens", 0))),
    ]


def demo_numbers() -> List[Tuple[str, str]]:
    """Demo 报告里的数字（README 的 demo 表格引用它们）。

    README 表格里用的是**简称**（如「AI工程」），所以这里也用简称比对——
    简称的唯一来源是文件名去掉「简历202607-」前缀，避免两处各写一套映射。
    """
    demo = ROOT / "experiments" / "results" / "demo" / "demo_run.json"
    if not demo.exists():
        return []
    data = json.loads(demo.read_text(encoding="utf-8"))
    out: List[Tuple[str, str]] = []
    for row in data.get("results", []):
        full = row["resume"]
        short = full.replace("简历202607-", "")
        out.append(
            (
                f"demo {short} 命中",
                f"| {short} | {row['hit']} | {row['partial']} | {row['missing']} | "
                f"{row['gaps']} | {row['rewrites']} | {row['questions']} |",
            )
        )
    return out


def main() -> int:
    if not README.exists():
        print("[错误] 找不到 README.md", file=sys.stderr)
        return 1
    readme = README.read_text(encoding="utf-8")
    numbers = expected_numbers() + demo_numbers()
    if not numbers:
        print("[错误] 找不到 eval_results.json，先跑 scripts/run_eval.py", file=sys.stderr)
        return 1

    missing: List[str] = []
    for label, value in numbers:
        if value not in readme:
            missing.append(f"{label}: {value}")

    print(f"核对 {len(numbers)} 个数字（来源：experiments/results/）")
    if missing:
        print("\n❌ README 里没有这些值（或值已过期）：")
        for item in missing:
            print(f"   - {item}")
        return 1

    print("✅ README 的数字与结果文件一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
