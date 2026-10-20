"""生成「简历 × 技能」的人工标注真值（data/annotations/resume_skill_truth.json）。

★ 为什么需要它：
「假绿」指标（该说缺失却报命中）必须有**独立于系统本身的真值**。
如果用系统自己的匹配逻辑去核对，等于自证——系统说命中，系统也说命中，
假绿数恒为 0，指标就废了。

所以这里做一份**人工标注**的「技能 × 简历」真值表：
对每份简历，逐个判断「这份简历是否具备该技能」。

★ 标注口径（写进产物里，可复核）：
- **具备**：简历里有该技能的**技术栈行**，或**带动作动词/量化结果的项目句**。
  （光出现名字不算——「了解 X」和「用 X 做了 Y」是两回事）
- **不具备**：简历里完全没有该技能的痕迹，或只出现在「了解/熟悉/学过」这类
  无产出表述里。

★ 标注范围：只覆盖**评测集里出现过的技能**（62 个），不穷举所有可能技能——
  那是无底洞，而且与本评测无关。

★ 技能名用**词表规范名**（jobfit.lexicon 的规范名），
  这样与系统输出、人工标注三方能对齐到同一集合。

标注者：本项目作者（一人标注，无第二标注者复核——这是已知边界，写进产物里）。

用法：
    python scripts/annotate_resume_truth.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jobfit.normalize import canonical  # noqa: E402

JD_DIR = ROOT / "data" / "jds"
RESUME_DIR = ROOT / "data" / "resumes"
OUT_DIR = ROOT / "data" / "annotations"


def all_annotated_skills() -> List[str]:
    """评测集里出现过的全部技能（归一化到规范名，去重、保持出现顺序）。"""
    seen: Dict[str, None] = {}
    for path in sorted(JD_DIR.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for req in payload["annotations"]["requirements"]:
            seen.setdefault(canonical(req["skill"]) or req["skill"], None)
    return list(seen)


#: 人工标注：简历名 -> **具备**的技能列表（其余一律「不具备」）
#:
#: ★ 只列「具备」的，语义更不容易出错——「不具备」是默认，
#:   标注者不用逐个确认「这条确实没有」，只需要确认「这条确实有」。
#:
#: ★ 判定依据（逐条可复核）：
#:   具备   —— 简历里有该技能的「技术栈行」或「带动作/量化的项目句」
#:   不具备 —— 完全没有该技能痕迹，或只出现在「了解/熟悉/学过」这类无产出表述里
HAVE: Dict[str, List[str]] = {
    "简历202607-AI工程": [
        "Python", "Java", "SQL", "Swift",
        "PyTorch", "Transformers", "HuggingFace", "NumPy", "Fastify", "Flask",
        "Spring Boot", "MyBatis", "Docker", "SwanLab", "Open3D",
        "Git", "Linux", "RESTful API", "工程化", "性能优化", "AI开发工具",
        "Word2Vec", "BERT", "Transformer", "文本分类", "图像超分辨率", "三维重建",
        "模型评估", "机器学习", "LLM微调", "Agent", "AI应用开发", "数据标注",
    ],
    "简历202607-Java开发": [
        "Python", "Java", "SQL", "Swift", "JavaScript",
        "PyTorch", "Transformers", "NumPy", "Fastify", "Spring Boot", "MyBatis",
        "SwanLab", "Open3D", "Vue", "MySQL", "Maven",
        "Git", "Linux", "RESTful API", "性能优化", "AI开发工具",
        "服务端开发", "数据库设计",
        "Word2Vec", "BERT", "Transformer", "文本分类", "图像超分辨率", "三维重建",
        "模型评估", "LLM微调",
    ],
    "简历202607-NLP算法": [
        "Python", "Java", "SQL", "Swift", "C++",
        "PyTorch", "Transformers", "HuggingFace", "Fastify", "Spring Boot",
        "SwanLab", "Open3D",
        "Git", "Linux", "工程化", "性能优化", "AI开发工具",
        "Word2Vec", "BERT", "Transformer", "文本分类", "图像超分辨率", "三维重建",
        "模型评估", "机器学习", "LLM微调", "Agent", "数据标注", "NLP",
    ],
    "简历202607-评测后训练-有联": [
        "Python", "Java", "SQL",
        "PyTorch", "Transformers", "HuggingFace", "NumPy", "SwanLab", "Open3D",
        "PEFT/LoRA",
        "Git", "Linux", "工程化", "性能优化", "AI开发工具", "模型部署", "部署",
        "Word2Vec", "BERT", "Transformer", "文本分类", "图像超分辨率", "三维重建",
        "模型评估", "机器学习", "LLM微调", "DPO", "后训练", "大模型评测",
        "评测指标体系", "数据标注", "NLP",
        # 自动化评测：「集成 SwanLab 实验管理平台，实现批量测试、自动打分、
        # 可视化报告一键生成」+「设计自动化评估管线」→ 明确具备。
        # （初版漏标，是标注错误，已修正；修正过程记在 git commit 里）
        "自动化评测",
    ],
}


def main() -> int:
    skills = all_annotated_skills()
    resume_names = sorted(p.stem for p in RESUME_DIR.glob("*.md"))
    print(f"评测集技能 {len(skills)} 个；简历 {len(resume_names)} 份")

    missing = [name for name in resume_names if name not in HAVE]
    if missing:
        print(f"[错误] 以下简历没有人工标注：{missing}", file=sys.stderr)
        return 1

    payload: Dict[str, Any] = {
        "rubric": {
            "具备": "简历里有该技能的技术栈行，或带动作动词/量化结果的项目句",
            "不具备": "完全没有该技能痕迹，或只出现在「了解/熟悉/学过」这类无产出表述里",
            "范围": "只标注评测集里出现过的技能，不穷举所有可能技能",
            "标注形式": "只列「具备」的技能，其余默认为「不具备」",
            "标注者": "本项目作者（一人标注，无第二标注者复核）",
            "已知偏差": (
                "「具备」的门槛带主观性——本表按「有技术栈行即算具备」处理，"
                "比严格的「必须要有项目级产出」宽松；这会让假绿指标偏保守"
                "（系统判命中而真值说不具备的情况会更少）。"
            ),
        },
        "skills": skills,
        "resumes": {},
    }

    for name in resume_names:
        have = {canonical(s) or s for s in HAVE[name]}
        # 只保留评测集里的技能（HAVE 里多写的技能与本评测无关，不进真值表）
        extra = sorted(have - set(skills))
        have = have & set(skills)
        payload["resumes"][name] = {skill: (skill in have) for skill in skills}
        if extra:
            print(f"[提示] {name} 标注里有 {len(extra)} 个技能不在评测集，已忽略：{extra}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / "resume_skill_truth.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(len(v) for v in payload["resumes"].values())
    positive = sum(1 for v in payload["resumes"].values() for x in v.values() if x)
    print(f"已写入 {target.relative_to(ROOT)}：{total} 个判定（具备 {positive} / 不具备 {total - positive}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
