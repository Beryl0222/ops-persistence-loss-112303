"""训练侧的可微持久度损失。

契约（见 README）：训练器提交要缩短或延长的 *有限持久配对*（用查询结果中的
完整配对身份：出生单元键 + 消失单元键）与权重；目标持久度 = 当前持久度 +
目标持久度差（延长为正、缩短为负，由调用方给符号）。损失为权重乘
``(持久度 - 目标持久度)²``。

梯度只返回影响出生与消失的原始格点分数，并按规范格点 (行, 列) 聚合——
多片重复声明的同一格点是同一个变量，不会重复计数。

三类允许训练跳过并给出原因：

* ``undetermined``：配对仍处于缺片未确定区域（未通过可证明性检查）；
* ``identity_mismatch``：请求的配对身份在当前过滤结果中不存在或已跳换；
* ``tie_risk``：同分变化可能导致配对跳换（严格模式下跳过，否则给次梯度并告警）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .cells import DIM_E, DIM_F, DIM_V, CellKey, edge_vertex_keys, face_corners
from .engine import PHResult, PersistencePair
from .observation import ComplexData

SKIP_UNDETERMINED = "undetermined"
SKIP_IDENTITY = "identity_mismatch"
SKIP_TIE = "tie_risk"
STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class TrainingRequest:
    """单个有限持久配对的训练请求。"""

    birth_key: CellKey
    death_key: CellKey
    target_delta: float          # 目标持久度差（带符号：延长正、缩短负）
    weight: float = 1.0
    loss_kind: str = "squared"   # squared（契约默认）或 hinge

    @classmethod
    def make(cls, pair_identity: Sequence, target_delta: float, weight: float = 1.0,
             loss_kind: str = "squared") -> "TrainingRequest":
        birth_key, death_key = pair_identity
        return cls(birth_key=tuple(birth_key), death_key=tuple(death_key),
                   target_delta=float(target_delta), weight=float(weight), loss_kind=loss_kind)


@dataclass(frozen=True)
class RequestOutcome:
    request: TrainingRequest
    status: str
    skip_reason: Optional[str]
    loss: float
    gradient: Tuple[Tuple[Tuple[int, int], float], ...]   # 规范格点 -> 梯度（已去重聚合）
    tie_warning: bool
    detail: str


@dataclass
class TrainingOutcome:
    total_loss: float
    outcomes: Tuple[RequestOutcome, ...]
    gradient_by_point: Dict[Tuple[int, int], float] = field(default_factory=dict)

    @property
    def skipped(self) -> Tuple[RequestOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status == STATUS_SKIPPED)

    def as_numpy(self):
        import numpy as np

        points = sorted(self.gradient_by_point)
        coords = np.array(points, dtype=np.int64) if points else np.zeros((0, 2), dtype=np.int64)
        grads = np.array([self.gradient_by_point[p] for p in points], dtype=np.float64)
        return coords, grads


def resolve_proposal(ph: PHResult, point: Tuple[int, int]) -> Tuple[PersistencePair, ...]:
    """把工艺员提案（消失格点）解析为候选 1 维配对：死亡面或代表链覆盖该格点者。

    有多个候选时全部返回，由工艺员选择；训练器只能使用选中后的完整配对身份。
    """
    cd = ph.complex_data
    candidates: List[PersistencePair] = []
    for pair in ph.pairs:
        if pair.dim != DIM_E or pair.infinite:
            continue
        if cd.cells[pair.death_idx].dim == DIM_F:
            corners = face_corners(cd.cells[pair.death_idx].key)
            if tuple(point) in corners:
                candidates.append(pair)
                continue
        if tuple(point) in ph.representative_points(pair):
            candidates.append(pair)
    return tuple(sorted(candidates, key=lambda p: (-p.persistence, p.birth_idx)))


def persistence_loss(
    ph: PHResult,
    requests: Sequence[TrainingRequest],
    strict_ties: bool = True,
) -> TrainingOutcome:
    """按契约计算可微持久度损失与原始格点梯度。"""
    cd = ph.complex_data
    outcomes: List[RequestOutcome] = []
    total = 0.0
    grad_acc: Dict[Tuple[int, int], float] = {}

    for request in requests:
        pair = ph.pair_for_identity(request.birth_key, request.death_key)
        if pair is None:
            # 身份缺失分两种语义：配对所在区域仍有缺片 -> 未定；否则才是配对跳换。
            if not ph.complex_data.fully_observed():
                outcomes.append(RequestOutcome(
                    request, STATUS_SKIPPED, SKIP_UNDETERMINED, 0.0, (), False,
                    "请求配对所在区域仍有缺片，身份暂不可解析，不得按跳换处理"))
            else:
                outcomes.append(RequestOutcome(
                    request, STATUS_SKIPPED, SKIP_IDENTITY, 0.0, (), False,
                    "请求的配对身份在当前过滤结果中不存在（可能已跳换）"))
            continue
        if not pair.certified:
            outcomes.append(RequestOutcome(request, STATUS_SKIPPED, SKIP_UNDETERMINED, 0.0, (),
                                           False, pair.uncertified_reason or "配对仍处于未确定区域"))
            continue
        tie_warning, tie_detail = _tie_risk(cd, pair)
        if tie_warning and strict_ties:
            outcomes.append(RequestOutcome(request, STATUS_SKIPPED, SKIP_TIE, 0.0, (),
                                           True, tie_detail))
            continue

        current = pair.persistence
        target = current + request.target_delta
        residual = current - target          # = -target_delta
        if request.loss_kind == "squared":
            loss = request.weight * residual * residual
            factor = 2.0 * request.weight * residual
        elif request.loss_kind == "hinge":
            # 延长请求：要求持久度至少达到目标；缩短请求：要求不超过目标。
            violated = residual > 0 if request.target_delta > 0 else residual < 0
            loss = request.weight * abs(residual) if violated else 0.0
            factor = request.weight * (1.0 if residual > 0 else -1.0) if violated else 0.0
        else:
            raise ValueError(f"未知损失类型：{request.loss_kind}")

        local_grad = _pair_gradient(cd, pair, factor)
        for point, value in local_grad:
            grad_acc[point] = grad_acc.get(point, 0.0) + value
        total += loss
        detail = tie_detail if tie_warning else "次梯度按固定同分裁决选择的关键单元计算"
        outcomes.append(RequestOutcome(
            request=request,
            status=STATUS_OK,
            skip_reason=None,
            loss=loss,
            gradient=tuple(sorted(local_grad)),
            tie_warning=tie_warning,
            detail=detail,
        ))

    return TrainingOutcome(total_loss=total, outcomes=tuple(outcomes), gradient_by_point=grad_acc)


# ----------------------------------------------------------------------
def _pair_gradient(cd: ComplexData, pair: PersistencePair, factor: float
                   ) -> Tuple[Tuple[Tuple[int, int], float], ...]:
    """返回 factor 乘 (d persistence / d score) 的格点梯度，按规范格点聚合。"""
    birth_cell = cd.cells[pair.birth_idx]
    death_cell = cd.cells[pair.death_idx]
    accum: Dict[Tuple[int, int], float] = {}

    def add(points: Sequence[Tuple[int, int]], sign: float) -> None:
        # argmax 选择由固定同分序唯一确定；同分时其余顶点梯度为 0（次梯度风险另行报告）。
        chosen = _argmax_point(cd, points)
        accum[chosen] = accum.get(chosen, 0.0) + sign * factor

    if birth_cell.dim == DIM_V:
        add([(birth_cell.row, birth_cell.col)], -1.0)
    else:
        add([_vertex_of_edge_key(birth_cell.key, 0), _vertex_of_edge_key(birth_cell.key, 1)], -1.0)

    if death_cell.dim == DIM_E:
        add([_vertex_of_edge_key(death_cell.key, 0), _vertex_of_edge_key(death_cell.key, 1)], +1.0)
    else:
        add(list(face_corners(death_cell.key)), +1.0)
    return tuple(sorted(accum.items()))


def _argmax_point(cd: ComplexData, points: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    def key(point: Tuple[int, int]):
        idx = cd.index[(DIM_V, point[0], point[1], 0)]
        return cd.cells[idx].value, point[0], point[1]

    return max(points, key=key)


def _vertex_of_edge_key(edge_key: CellKey, which: int) -> Tuple[int, int]:
    a, b = edge_vertex_keys(edge_key)
    key = (a, b)[which]
    return key[1], key[2]


def _tie_risk(cd: ComplexData, pair: PersistencePair) -> Tuple[bool, str]:
    """关键单元的 argmax/argmin 是否存在同分并列（同分变化可能导致配对跳换）。"""
    messages: List[str] = []
    birth_cell = cd.cells[pair.birth_idx]
    death_cell = cd.cells[pair.death_idx]

    def check_max(points: Sequence[Tuple[int, int]], label: str) -> None:
        values = [cd.cells[cd.index[(DIM_V, r, c, 0)]].value for r, c in points]
        top = max(values)
        if sum(1 for value in values if value == top) > 1:
            tied = [p for p, v in zip(points, values) if v == top]
            messages.append(f"{label} 的最大值在 {tied} 同分并列")

    if birth_cell.dim == DIM_E:
        check_max([_vertex_of_edge_key(birth_cell.key, 0),
                   _vertex_of_edge_key(birth_cell.key, 1)], "出生边过滤值")
    if death_cell.dim == DIM_F:
        check_max(list(face_corners(death_cell.key)), "消失面过滤值")
    elif death_cell.dim == DIM_E:
        check_max([_vertex_of_edge_key(death_cell.key, 0),
                   _vertex_of_edge_key(death_cell.key, 1)], "消失边过滤值")
    return bool(messages), "；".join(messages)
