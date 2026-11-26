"""F₂ 上的分维块持久同调引擎：0D/1D 配对、代表循环、工作预算与可证明性。

2D 方形胞腔复形上把边界矩阵按维块分解（等价于标准 Smith 化简，但避免
维块混用带来的主元错误）：

1. **0 维块**：边列对顶点列消元。消不空的边与顶点配对（连通分量合并）；
   消成空的边列即 1-环的出生边，其化简后的边链是代表环；
2. **1 维块**：只取环边作为坐标，把每个面的边界投影到环边空间，再做左到右
   消元。获得主元（环边）的面与该环边配对（孔洞闭合），其化简链是孔洞的
   代表循环；最终仍为空的环边给出无限 1 维条。

可证明性（缺片语义）：

* 已完成的有限配对是持久同调不变量，只可能被“涉及观测前沿单元”的补数改变；
  配对两端及其代表链全部落在内部（不邻接任何缺片格点）时出具 ``certified``；
* 无限条只有在复形封口（窗口内没有缺片）且代表链内部时才可证明；
* 未证明的条一律标为未定，不得把缺片临时边界当成孔洞。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .cells import DIM_E, DIM_F, DIM_V, CellKey
from .observation import ComplexData

INF = float("inf")


@dataclass(frozen=True)
class PersistencePair:
    """一条持久条 / 一个完整配对身份。"""

    dim: int
    birth_idx: int
    death_idx: int                  # 无限条的死亡下标为 -1
    birth_value: float
    death_value: float
    persistence: float
    certified: bool
    representative: Tuple[int, ...] = ()     # 1 维条的代表边链（下标）
    uncertified_reason: Optional[str] = None

    @property
    def infinite(self) -> bool:
        return self.death_idx < 0

    def identity(self, complex_data: ComplexData) -> Tuple[CellKey, Optional[CellKey]]:
        """训练/台账使用的完整配对身份：(出生单元键, 消失单元键)。"""
        birth_key = complex_data.cells[self.birth_idx].key
        death_key = None if self.infinite else complex_data.cells[self.death_idx].key
        return birth_key, death_key


@dataclass
class PHResult:
    pairs: Tuple[PersistencePair, ...]
    complex_data: ComplexData
    complete: bool
    work_used: int
    budget: Optional[int]
    pairs_by_birth: Dict[int, PersistencePair] = field(default_factory=dict)

    def finite_pairs(self, dim: Optional[int] = None) -> Tuple[PersistencePair, ...]:
        out = (p for p in self.pairs if not p.infinite and (dim is None or p.dim == dim))
        return tuple(sorted(out, key=lambda p: (-p.persistence, p.birth_idx)))

    def infinite_bars(self, dim: Optional[int] = None) -> Tuple[PersistencePair, ...]:
        return tuple(p for p in self.pairs if p.infinite and (dim is None or p.dim == dim))

    def pair_for_identity(self, birth_key: CellKey, death_key: Optional[CellKey]) -> Optional[PersistencePair]:
        cd = self.complex_data
        birth_idx = cd.index.get(birth_key)
        if birth_idx is None:
            return None
        pair = self.pairs_by_birth.get(birth_idx)
        if pair is None:
            return None
        actual_death = None if pair.infinite else cd.cells[pair.death_idx].key
        return pair if actual_death == death_key else None

    def representative_points(self, pair: PersistencePair) -> Tuple[Tuple[int, int], ...]:
        """代表边链经过的全部顶点（行、列），供跨区路径展示。"""
        from .cells import edge_endpoints

        points: List[Tuple[int, int]] = []
        seen = set()
        for edge_idx in pair.representative:
            for point in edge_endpoints(self.complex_data.cells[edge_idx].key):
                if point not in seen:
                    seen.add(point)
                    points.append(point)
        return tuple(points)


class ReductionEngine:
    """可恢复的分维块化简器（预算耗尽后可用更大预算继续）。"""

    def __init__(self, complex_data: ComplexData):
        self.cd = complex_data
        n = len(complex_data.cells)
        self.work_used = 0
        self.complete = False
        # 0 维块结果
        self.edge_reduced: Dict[int, List[int]] = {}   # 边列 -> 化简后的顶点链
        self.cycle_edges: List[int] = []               # 消成空的边（1-环出生）
        self.cycle_birth_chain: Dict[int, Tuple[int, ...]] = {}  # 环边 -> 代表环边链
        self.h0_pairs: List[Tuple[int, int]] = []
        # 1 维块结果
        self.h1_pairs: List[Tuple[int, int, Tuple[int, ...]]] = []
        self.infinite_edge_chain: Dict[int, Tuple[int, ...]] = {}

    # ------------------------------------------------------------------
    def reduce(self, budget: Optional[int] = None) -> PHResult:
        cd = self.cd
        n = len(cd.cells)
        vertex_pivot: Dict[int, int] = {}   # 顶点主元 -> 供体边列
        edge_cursor = 0

        # ---- 0 维块：边列对顶点主元消元；同时跟踪每条边在边空间中的 1-链 ----
        edge_cols = [j for j, cell in enumerate(cd.cells) if cell.dim == DIM_E and j < n]
        edge_chain: Dict[int, List[int]] = {j: [j] for j in edge_cols}
        while edge_cursor < len(edge_cols):
            j = edge_cols[edge_cursor]
            column = list(cd.boundaries[j])
            chain = list(edge_chain[j])
            while column:
                pivot = column[-1]
                donor = vertex_pivot.get(pivot)
                if donor is None:
                    vertex_pivot[pivot] = j
                    break
                column = _xor_sorted(column, self.edge_reduced[donor])
                chain = _xor_sorted(chain, edge_chain[donor])
                self.work_used += len(self.edge_reduced[donor])
                if budget is not None and self.work_used > budget:
                    return self._partial_result(budget)
            self.edge_reduced[j] = column
            edge_chain[j] = chain
            if column:
                self.h0_pairs.append((column[-1], j))
            else:
                self.cycle_edges.append(j)
                self.cycle_birth_chain[j] = tuple(chain)
            edge_cursor += 1

        # ---- 1 维块：面边界投影到环边空间，再消元 ----
        cycle_index = {edge_idx: k for k, edge_idx in enumerate(self.cycle_edges)}
        edge_pivot: Dict[int, int] = {}       # 环边主元（环边坐标）-> 供体面列
        face_cols = [j for j, cell in enumerate(cd.cells) if cell.dim == DIM_F]
        face_reduced: Dict[int, List[int]] = {}
        for j in face_cols:
            # 面的边界边中，只有环边坐标参与 1 维块（非环边已在 0 维块被张成）。
            projected = sorted(
                cycle_index[e] for e in cd.boundaries[j] if e in cycle_index
            )
            column = projected
            while column:
                pivot = column[-1]
                donor = edge_pivot.get(pivot)
                if donor is None:
                    edge_pivot[pivot] = j
                    break
                column = _xor_sorted(column, face_reduced[donor])
                self.work_used += len(face_reduced[donor])
                if budget is not None and self.work_used > budget:
                    return self._partial_result(budget)
            face_reduced[j] = column
            if column:
                birth_edge = self.cycle_edges[column[-1]]
                # 代表循环 = 化简链中各环边的出生环链在边空间中的异或。
                chain_set: List[int] = []
                for cycle_coord in column:
                    chain_set = _xor_sorted(
                        chain_set, list(self.cycle_birth_chain[self.cycle_edges[cycle_coord]])
                    )
                self.h1_pairs.append((birth_edge, j, tuple(chain_set)))
            # 消成空的面：该面只张成已有环，不产生配对。

        alive_edges = {birth for birth, _, _ in self.h1_pairs}
        for edge_idx in self.cycle_edges:
            if edge_idx not in alive_edges:
                self.infinite_edge_chain[edge_idx] = self.cycle_birth_chain[edge_idx]

        self.complete = True
        return self._build_result(budget)

    # ------------------------------------------------------------------
    def _partial_result(self, budget: Optional[int]) -> PHResult:
        # 预算耗尽：只输出已完成配对，complete=False 让调用方走“可证明部分”路径。
        return self._build_result(budget, complete=False)

    def _build_result(self, budget: Optional[int], complete: bool = True) -> PHResult:
        cd = self.cd
        frontier = cd.frontier
        dsu = _DUnion(cd)
        comp_touches = dsu.component_frontier_marks(frontier)
        sealed = not cd.unknown_vertices
        pairs: List[PersistencePair] = []
        by_birth: Dict[int, PersistencePair] = {}

        def add(pair: PersistencePair) -> None:
            pairs.append(pair)
            by_birth[pair.birth_idx] = pair

        # 0 维有限配对与无限 0 维条
        paired_vertices = {birth for birth, _ in self.h0_pairs}
        for birth, death in self.h0_pairs:
            add(self._h0_finite(birth, death, frontier))
        for j, cell in enumerate(cd.cells):
            if cell.dim == DIM_V and j not in paired_vertices:
                certified = sealed and not comp_touches.get(dsu.find(j), False)
                reason = None if certified else "无限 0 维条接触观测前沿，后续补片可能引发合并"
                add(PersistencePair(
                    dim=0, birth_idx=j, death_idx=-1, birth_value=cell.value,
                    death_value=INF, persistence=INF, certified=certified,
                    representative=(), uncertified_reason=reason,
                ))

        # 1 维有限配对
        for birth, death, chain in self.h1_pairs:
            on_frontier = sorted({i for i in (birth, death, *chain) if i in frontier})
            certified = not on_frontier
            reason = None
            if not certified:
                keys = ", ".join(str(cd.cells[i].key) for i in on_frontier)
                reason = f"配对单元或代表链接触观测前沿：{keys}"
            add(PersistencePair(
                dim=1, birth_idx=birth, death_idx=death,
                birth_value=cd.cells[birth].value, death_value=cd.cells[death].value,
                persistence=cd.cells[death].value - cd.cells[birth].value,
                certified=certified, representative=chain, uncertified_reason=reason,
            ))

        # 1 维无限条只有在 1 维块完整跑完时才能下结论；预算中断时保持未定。
        if complete:
            for edge_idx, chain in self.infinite_edge_chain.items():
                touches = any(i in frontier for i in chain)
                certified = sealed and not touches
                if not sealed:
                    reason = "窗口未封口（存在缺片），1 维无限条可能被后续面片闭合"
                elif touches:
                    reason = "代表循环接触观测前沿"
                else:
                    reason = None
                add(PersistencePair(
                    dim=1, birth_idx=edge_idx, death_idx=-1,
                    birth_value=cd.cells[edge_idx].value, death_value=INF,
                    persistence=INF, certified=certified, representative=chain,
                    uncertified_reason=reason,
                ))

        return PHResult(
            pairs=tuple(pairs), complex_data=cd, complete=complete,
            work_used=self.work_used, budget=budget, pairs_by_birth=by_birth,
        )

    def _h0_finite(self, birth: int, death: int, frontier: frozenset) -> PersistencePair:
        cd = self.cd
        on_frontier = sorted({i for i in (birth, death) if i in frontier})
        certified = not on_frontier
        reason = None
        if not certified:
            keys = ", ".join(str(cd.cells[i].key) for i in on_frontier)
            reason = f"配对单元接触观测前沿：{keys}"
        return PersistencePair(
            dim=0, birth_idx=birth, death_idx=death,
            birth_value=cd.cells[birth].value, death_value=cd.cells[death].value,
            persistence=cd.cells[death].value - cd.cells[birth].value,
            certified=certified, representative=(), uncertified_reason=reason,
        )


def run_persistence(complex_data: ComplexData, budget: Optional[int] = None) -> PHResult:
    """计算持久配对；``budget`` 给定时只保证预算内已完成的部分。"""
    return ReductionEngine(complex_data).reduce(budget)


def _xor_sorted(a: Sequence[int], b: Sequence[int]) -> List[int]:
    i = j = 0
    out: List[int] = []
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] > b[j]:
            out.append(b[j])
            j += 1
        else:
            i += 1
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


class _DUnion:
    """由 0/1 骨架边构建的并查集，用于判断无限 0 维条的稳定性。"""

    def __init__(self, cd: ComplexData):
        self.parent = list(range(len(cd.cells)))
        for j, cell in enumerate(cd.cells):
            if cell.dim == DIM_E:
                a, b = cd.boundaries[j]
                self.union(a, b)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def component_frontier_marks(self, frontier: frozenset) -> Dict[int, bool]:
        marks: Dict[int, bool] = {}
        for idx in frontier:
            marks[self.find(idx)] = True
        return marks
