"""评测指标计算：抽取 P/R/F1、证据定位正确率、假绿计数。

★ 三个口径必须写死在这里（README 与报告都引用这里的定义）：

**1. 抽取 P/R/F1（按项）**
- 预测集 = 抽取器给出的技能名集合（归一化后）
- 真值集 = 人工标注的技能名集合（归一化后）
- 命中 = 归一化后完全相同（不做模糊匹配——模糊匹配会把
  「Spring Boot」和「Spring Cloud」算成一个，那就不是抽取准确率了）
- P = TP / |预测|，R = TP / |真值|，F1 = 2PR/(P+R)

**2. 重要度加权 F1**
- 权重：必须 = 2.0，加分 = 1.0
- 加权 TP = 命中项的**真值**权重之和（真值的权重才是「这条要求有多重要」）
- 加权 FP = 预测项（未命中）的**预测**权重之和
- 加权 FN = 真值项（未命中）的**真值**权重之和
- 加权 P = wTP / (wTP + wFP)，加权 R = wTP / (wTP + wFN)

**3. 证据定位正确率**
两条独立的检查，都必须过：
- (a) **证据在简历里**：证据句是简历原文的子串（去空白后比对）
- (b) **证据支持结论**：证据句里确实出现了该技能
两个都过才算「正确定位」。分母是所有「命中」+「部分命中」的结论
（缺失的结论没有证据，不参与）。

**4. 假绿（false green）**
定义：该判「缺失」却报了「命中」的条数。
判据用**独立于匹配器**的复核：拿人工标注的技能要求重新核对
——若某要求在人工标注里被标为该简历「不具备」，而系统报「命中」，即为假绿。
这需要每条要求 × 每份简历有「是否具备」的人工判断；
本项目的做法是：用「简历里是否出现该技能 + 是否有项目级证据」做**第二次独立计算**
（scripts/run_eval.py 里的 reference_match），它与匹配器用不同的代码路径，
因此可以作为交叉核对。**不能只用匹配器自己算自己**——那等于自证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from jobfit.normalize import match_key
from jobfit.schemas import IMPORTANCE_MUST, IMPORTANCE_PLUS


def _key_hits(pred_key: str, truth_key: str) -> bool:
    """两个键是否指向同一个技能。

    复合短语会产生多键（`a|b`）。判定规则：
    - 单键 vs 单键：完全相同才算命中；
    - 任一方是多键：**有一键重合**即算命中
      （「Agent、工具调用」应能对上标注里的「Agent」或「工具调用」）。
    """
    if pred_key == truth_key:
        return True
    pred_parts = set(pred_key.split("|"))
    truth_parts = set(truth_key.split("|"))
    return bool(pred_parts & truth_parts)


@dataclass
class Prf:
    """一组 P/R/F1 数字。"""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    weighted_tp: float = 0.0
    weighted_fp: float = 0.0
    weighted_fn: float = 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return round(self.tp / denom, 4) if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return round(self.tp / denom, 4) if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    @property
    def weighted_precision(self) -> float:
        denom = self.weighted_tp + self.weighted_fp
        return round(self.weighted_tp / denom, 4) if denom else 0.0

    @property
    def weighted_recall(self) -> float:
        denom = self.weighted_tp + self.weighted_fn
        return round(self.weighted_tp / denom, 4) if denom else 0.0

    @property
    def weighted_f1(self) -> float:
        p, r = self.weighted_precision, self.weighted_recall
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "weighted_tp": round(self.weighted_tp, 4),
            "weighted_fp": round(self.weighted_fp, 4),
            "weighted_fn": round(self.weighted_fn, 4),
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
        }


def _weight(importance: str) -> float:
    return 2.0 if importance == IMPORTANCE_MUST else 1.0


def score_extraction(
    predicted: Sequence[Tuple[str, str]],
    truth: Sequence[Tuple[str, str]],
) -> Tuple[Prf, List[Dict[str, Any]]]:
    """算一组抽取结果的 P/R/F1。

    `predicted` / `truth` 都是 [(技能名, 重要度), ...]。
    返回 (指标, 逐项明细)。

    ★ 匹配用 `match_key`（归一化到词表规范名 + 复合键重叠判定），
      所以「SFT」能对上「LLM微调」、「RLVR」能对上「DPO」。
    """
    pred_items: List[Tuple[str, str, str]] = []  # (原始名, 键, 重要度)
    truth_items: List[Tuple[str, str, str]] = []
    seen_pred: Set[str] = set()
    seen_truth: Set[str] = set()
    for name, importance in predicted:
        key = match_key(name)
        if not key or key in seen_pred:
            continue
        seen_pred.add(key)
        pred_items.append((name, key, importance))
    for name, importance in truth:
        key = match_key(name)
        if not key or key in seen_truth:
            continue
        seen_truth.add(key)
        truth_items.append((name, key, importance))

    prf = Prf()
    detail: List[Dict[str, Any]] = []

    matched_pred: Set[int] = set()
    matched_truth: Set[int] = set()

    # 先按真值顺序找命中（保证「同一真值不被两条预测重复占用」）
    for ti, (truth_name, truth_key, truth_imp) in enumerate(truth_items):
        for pi, (pred_name, pred_key, _) in enumerate(pred_items):
            if pi in matched_pred:
                continue
            if _key_hits(pred_key, truth_key):
                matched_pred.add(pi)
                matched_truth.add(ti)
                prf.tp += 1
                weight = _weight(truth_imp)
                prf.weighted_tp += weight
                detail.append(
                    {
                        "skill": truth_name,
                        "kind": "命中",
                        "predicted": pred_name,
                        "predicted_importance": pred_items[pi][2],
                        "truth_importance": truth_imp,
                    }
                )
                break

    # 未命中的预测 → 误报
    for pi, (pred_name, pred_key, pred_imp) in enumerate(pred_items):
        if pi in matched_pred:
            continue
        prf.fp += 1
        weight = _weight(pred_imp)
        prf.weighted_fp += weight
        detail.append(
            {
                "skill": pred_name,
                "kind": "误报",
                "predicted": pred_name,
                "predicted_importance": pred_imp,
                "truth_importance": "",
            }
        )

    # 未命中的真值 → 漏报
    for ti, (truth_name, truth_key, truth_imp) in enumerate(truth_items):
        if ti in matched_truth:
            continue
        prf.fn += 1
        weight = _weight(truth_imp)
        prf.weighted_fn += weight
        detail.append(
            {
                "skill": truth_name,
                "kind": "漏报",
                "predicted": "",
                "predicted_importance": "",
                "truth_importance": truth_imp,
            }
        )

    return prf, detail


def aggregate_prf(items: Sequence[Prf]) -> Prf:
    """把多组（每份 JD 一组）的指标按**计数**汇总成总体指标。

    注意不是「先算每份的 F1 再平均」——那是 macro F1，
    对「有 8 条要求的 JD」和「有 14 条要求的 JD」等权，
    会让小 JD 的噪声被放大。评测报告两个都给（micro 与 macro）。
    """
    total = Prf()
    for item in items:
        total.tp += item.tp
        total.fp += item.fp
        total.fn += item.fn
        total.weighted_tp += item.weighted_tp
        total.weighted_fp += item.weighted_fp
        total.weighted_fn += item.weighted_fn
    return total


def macro_f1(items: Sequence[Prf]) -> Dict[str, float]:
    """macro 平均（每份 JD 等权）。"""
    if not items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "weighted_precision": 0.0, "weighted_recall": 0.0, "weighted_f1": 0.0}
    n = len(items)
    return {
        "precision": round(sum(i.precision for i in items) / n, 4),
        "recall": round(sum(i.recall for i in items) / n, 4),
        "f1": round(sum(i.f1 for i in items) / n, 4),
        "weighted_precision": round(sum(i.weighted_precision for i in items) / n, 4),
        "weighted_recall": round(sum(i.weighted_recall for i in items) / n, 4),
        "weighted_f1": round(sum(i.weighted_f1 for i in items) / n, 4),
    }


@dataclass
class EvidenceCheck:
    """证据定位核对结果。"""

    total: int = 0
    in_resume: int = 0
    supports_skill: int = 0
    both: int = 0
    failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return round(self.both / self.total, 4) if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "evidence_in_resume": self.in_resume,
            "evidence_supports_skill": self.supports_skill,
            "both_correct": self.both,
            "accuracy": self.accuracy,
            "failures": self.failures[:20],
        }


def check_evidence(
    matches: Sequence[Any],
    resume_text: str,
    is_verbatim: Any,
    supports: Any,
) -> EvidenceCheck:
    """核对每条结论的证据。

    `is_verbatim(evidence, resume_text)` 与 `supports(evidence, skill)`
    由调用方注入——这样核对逻辑与匹配器**走不同代码路径**，
    避免「用同一份代码自证正确」。
    """
    check = EvidenceCheck()
    for match in matches:
        if match.status == "缺失":
            continue  # 缺失没有证据，不参与
        evidence = (match.evidence or "").strip()
        check.total += 1
        in_resume = bool(evidence) and bool(is_verbatim(evidence, resume_text))
        supports_skill = bool(evidence) and bool(supports(evidence, match.requirement.skill))
        if in_resume:
            check.in_resume += 1
        if supports_skill:
            check.supports_skill += 1
        if in_resume and supports_skill:
            check.both += 1
        else:
            check.failures.append(
                {
                    "skill": match.requirement.skill,
                    "status": match.status,
                    "evidence": evidence[:120],
                    "in_resume": in_resume,
                    "supports_skill": supports_skill,
                }
            )
    return check


def count_false_green(
    system_matches: Sequence[Any],
    reference_missing: Iterable[str],
) -> List[Dict[str, Any]]:
    """假绿计数：系统报「命中」但独立参考判定为「不具备」的条数。

    `reference_missing` 是独立参考算出来的「该简历不具备」的技能集合。
    """
    missing = set(reference_missing)
    false_green: List[Dict[str, Any]] = []
    missing_keys = {match_key(m) for m in missing}
    for match in system_matches:
        if match.status != "命中":
            continue
        if match_key(match.requirement.skill) in missing_keys:
            false_green.append(
                {
                    "skill": match.requirement.skill,
                    "evidence": (match.evidence or "")[:120],
                    "note": "系统判命中，人工标注真值判不具备",
                }
            )
    return false_green
