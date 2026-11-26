"""服务门面：全工件、裁取与条带化复核，以及换图兼容性检查。

条带化只解决内存/工作预算：窗口带一行 halo，内部单元（不邻接窗口边界）
上的持久配对与全量化简逐对相同；跨窗配对留待后续窗口或最终窗口证明。
未封口窗口的无限条一律不可证明，不会把临时边界当成孔洞。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .cells import DIM_E, CellKey, edge_endpoints
from .engine import PHResult, PersistencePair, run_persistence
from .observation import ComplexData, ObservationModel
from .review import (
    BudgetExhausted,
    ReviewLedger,
    ReviewOrder,
    review_region,
    review_workpiece,
)
from .training import TrainingOutcome, TrainingRequest, persistence_loss, resolve_proposal

Box = Tuple[int, int, int, int]


@dataclass
class StripeResult:
    stripe_index: int
    box: Box
    ph: PHResult
    certified_pairs: Tuple[PersistencePair, ...]
    pending_pair_keys: Tuple[Tuple[CellKey, Optional[CellKey]], ...]


class TopoService:
    """同一份分块打分同时服务工件复核与训练损失的统一入口。"""

    def __init__(self, model: ObservationModel):
        self.model = model
        self.ledger = ReviewLedger()

    # ------------------------------------------------------------------
    # 复核
    # ------------------------------------------------------------------
    def workpiece_review(self, orders: Mapping[str, ReviewOrder],
                         budget: Optional[int] = None) -> Tuple[List, PHResult]:
        entries, ph = review_workpiece(self.model, orders, budget=budget)
        self.ledger.update(entries)
        return entries, ph

    def region_review(self, order: ReviewOrder,
                      budget: Optional[int] = None) -> Tuple[List, PHResult]:
        entries, ph = review_region(self.model, order, budget=budget)
        self.ledger.update(entries)
        return entries, ph

    def crop_review(self, box: Box, persistence_threshold: float,
                    score_range: Optional[Tuple[float, float]] = None,
                    budget: Optional[int] = None) -> Tuple[List, PHResult]:
        """不改变复核范围边界语义的裁取复核。"""
        order = ReviewOrder(region_id=f"crop{box}", box=tuple(box),
                            persistence_threshold=persistence_threshold,
                            score_range=score_range)
        return self.region_review(order, budget=budget)

    def ingest_update(self, new_model: ObservationModel,
                      orders: Mapping[str, ReviewOrder]) -> List[Dict[str, str]]:
        """收到新分片后的替代：只有旧 pending 可被新可证明结论替代。"""
        if new_model.shape != self.model.shape:
            raise ValueError("换图尺寸不兼容")
        self.model = new_model
        entries, _ = review_workpiece(new_model, orders)
        return self.ledger.update(entries)

    # ------------------------------------------------------------------
    # 条带化（内存/预算约束）
    # ------------------------------------------------------------------
    def stripe_review(self, stripe_rows: int, halo: int = 2) -> Tuple[StripeResult, ...]:
        """按水平条带分窗计算；``halo`` 行保证跨面环在窗内完整。

        认证规则（同时满足才可证明，且与全量化简逐对相同）：

        * 配对两端及代表链均不触观测前沿（缺片语义）；
        * 配对两端不在窗口边界单元集合（跨窗边及其入射面）上；
        * 代表链经过的顶点全部落在内部行区间。

        跨窗配对留待内部包含该面的后续窗口；未封口窗口的无限条一律不可证明，
        不会把临时窗口边界当成孔洞。
        """
        if stripe_rows <= 1:
            raise ValueError("条带高度至少 2 行，才能容纳边和面")
        if halo < 1:
            raise ValueError("halo 至少 1 行；跨面环建议 2 行")
        rows, cols = self.model.shape
        results: List[StripeResult] = []
        start = 0
        stripe_index = 0
        while start < rows:
            end = min(rows, start + stripe_rows)
            is_last = end >= rows
            box = (max(0, start - halo), 0, min(rows, end + halo), cols)
            cd = self.model.build_complex(box=box)
            ph = run_persistence(cd)
            boundary = self._window_boundary(cd, start, end)
            certified, pending = self._certified_in_interior(
                ph, start, end, is_last, boundary
            )
            results.append(StripeResult(stripe_index, box, ph, certified, pending))
            stripe_index += 1
            if is_last:
                break
            start = end
        return tuple(results)

    @staticmethod
    def _window_boundary(cd: ComplexData, interior_r0: int, interior_r1: int) -> frozenset:
        """跨越内部行界的边，以及与这些边共边的面。

        这些单元的配对可能依赖窗外数据；内部面只要不在此集合，其完整闭星与
        环邻域都已被 halo 覆盖，配对与全量结果相同。
        """
        boundary: set = set()
        for idx, cell in enumerate(cd.cells):
            if cell.dim != DIM_E:
                continue
            (ar, _), (br, _) = edge_endpoints(cell.key)
            if (ar == interior_r0 and br == interior_r0 - 1) or (
                ar == interior_r1 - 1 and br == interior_r1
            ):
                boundary.add(idx)
        edge_boundary = set(boundary)
        for face_idx, cell in enumerate(cd.cells):
            if cell.dim != 2:
                continue
            if any(edge_idx in edge_boundary for edge_idx in cd.boundaries[face_idx]):
                boundary.add(face_idx)
        return frozenset(boundary)

    def _certified_in_interior(self, ph: PHResult, interior_r0: int,
                               interior_r1: int, is_last: bool,
                               window_boundary: frozenset):
        cd = ph.complex_data
        certified: List[PersistencePair] = []
        pending: List[Tuple[CellKey, Optional[CellKey]]] = []
        for pair in ph.pairs:
            identity = pair.identity(cd)
            points = ph.representative_points(pair) if pair.dim == DIM_E else ()
            # 代表链顶点必须落在 halo 窗口内（构建时已保证）；这里额外要求链不触窗边界。
            inside = all(cd.box[0] <= r < cd.box[2] for r, _ in points)
            if pair.infinite:
                # 无限条只有最终封口窗口、链不触窗边界且不触观测前沿时可证明。
                chain_clean = all(idx not in window_boundary for idx in pair.representative)
                if is_last and inside and chain_clean and pair.certified:
                    certified.append(pair)
                else:
                    pending.append(identity)
                continue
            involved = (pair.birth_idx, pair.death_idx, *pair.representative)
            if (
                all(not cd.is_frontier(idx) for idx in involved)
                and all(idx not in window_boundary for idx in involved)
            ):
                certified.append(pair)
            else:
                pending.append(identity)
        return tuple(certified), tuple(pending)

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def full_persistence(self, budget: Optional[int] = None) -> PHResult:
        cd = self.model.build_complex()
        ph = run_persistence(cd, budget=budget)
        if not ph.complete:
            raise BudgetExhausted(ph.work_used)
        return ph

    def resolve_proposal(self, point: Tuple[int, int], ph: Optional[PHResult] = None
                         ) -> Tuple[PersistencePair, ...]:
        return resolve_proposal(ph or self.full_persistence(), point)

    def training_loss(self, requests: Sequence[TrainingRequest],
                      strict_ties: bool = True, ph: Optional[PHResult] = None) -> TrainingOutcome:
        return persistence_loss(ph or self.full_persistence(), requests, strict_ties=strict_ties)

    # ------------------------------------------------------------------
    # 换图兼容性
    # ------------------------------------------------------------------
    @staticmethod
    def check_compatible(model_a: ObservationModel, model_b: ObservationModel) -> None:
        """兼容图：同尺寸、同永久开孔、已观测格点的规范分数与计分身份一致。"""
        if model_a.shape != model_b.shape:
            raise ValueError("曲面尺寸不同，无法换图查询")
        if model_a.permanent_holes != model_b.permanent_holes:
            raise ValueError("永久开孔几何不同，不属于兼容图")
        rows, cols = model_a.shape
        for r in range(rows):
            for c in range(cols):
                sa, sb = model_a.score(r, c), model_b.score(r, c)
                if sa is None or sb is None:
                    continue
                if sa != sb:
                    raise ValueError(f"格点 ({r},{c}) 分数不一致：{sa} vs {sb}")
                if model_a.point_tile_ids(r, c) != model_b.point_tile_ids(r, c):
                    raise ValueError(f"格点 ({r},{c}) 计分身份不一致")
