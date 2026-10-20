"""job-fit-assistant：求职助手（AI 应用落地项目）。

六步流水线：
    JD 技能要求抽取 → 简历解析 → 匹配分析 → 差距与优先级建议 → 简历改写建议 → 预测面试题

CLI：
    python -m jobfit analyze --jd <file> --resume <file> [--format md|json]
    python -m jobfit serve
    python -m jobfit doctor
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
