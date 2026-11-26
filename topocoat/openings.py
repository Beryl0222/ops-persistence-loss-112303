"""工艺允许开孔的同调等价判定。

规则（见 README）：永久开孔面被永久去掉、四条边保留；工艺允许循环是 *开孔边界
生成的同调子空间*。比较必须使用满过滤范围中的类，不能按出生点是否靠近开孔判断。

实现：把窗口内可完整生效的开孔面补回，构建增广复形并做 F₂ 持久化简。
一条原始 1-环在增广复形中：

* 被某张补回的开孔面闭合 -> 它与开孔边界同调等价，放行；
* 仍是无限条或被普通面闭合 -> 不属于允许子空间。

判定对“开孔边界 + 同调路径”形式的代表环同样成立（这是同调等价而非逐边相等）。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .cells import DIM_E, DIM_F, DIM_V, Cell, CellKey, edge_vertex_keys, face_boundary_keys, face_corners, filtration_order
from .engine import PHResult, ReductionEngine
from .observation import ComplexData


def opening_faces_in_window(complex_data: ComplexData) -> Tuple[CellKey, ...]:
    """窗口内四个角点都在域内、可补回的永久开孔面键。"""
    cd = complex_data
    r0, c0, r1, c1 = cd.box
    keys: List[CellKey] = []
    for hr, hc in sorted(cd.permanent_holes):
        face_key = (DIM_F, hr, hc, 0)
        if face_key in cd.index:
            continue
        if not (r0 <= hr and c0 <= hc and hr + 1 < r1 and hc + 1 < c1):
            continue
        if all((DIM_V, rr, cc, 0) in cd.index for rr, cc in face_corners(face_key)):
            keys.append(face_key)
    return tuple(keys)


def augmented_complex_data(complex_data: ComplexData) -> ComplexData:
    """把永久开孔面补回得到增广复形（新增面按固定同分序插入）。"""
    cd = complex_data
    extra: List[Cell] = []
    for face_key in opening_faces_in_window(cd):
        value = max(
            cd.cells[cd.index[(DIM_V, rr, cc, 0)]].value
            for rr, cc in face_corners(face_key)
        )
        extra.append(Cell(key=face_key, value=value))
    if not extra:
        return cd

    ordered = filtration_order(list(cd.cells) + extra)
    index = {cell.key: i for i, cell in enumerate(ordered)}
    old_prov = {cell.key: prov for cell, prov in zip(cd.cells, cd.provenance)}
    boundaries: List[Tuple[int, ...]] = []
    provenance: List[Tuple[str, ...]] = []
    for cell in ordered:
        if cell.dim == DIM_V:
            boundaries.append(())
        elif cell.dim == DIM_E:
            boundaries.append(tuple(sorted(index[k] for k in edge_vertex_keys(cell.key))))
        else:
            boundaries.append(tuple(sorted(index[k] for k in face_boundary_keys(cell.key))))
        provenance.append(old_prov.get(cell.key, ("__opening__",)))

    return ComplexData(
        cells=tuple(ordered),
        index=index,
        boundaries=tuple(boundaries),
        frontier=cd.frontier,
        unknown_vertices=cd.unknown_vertices,
        provenance=tuple(provenance),
        shape=cd.shape,
        box=cd.box,
        permanent_holes=cd.permanent_holes,
    )


def allowed_cycle_births(complex_data: ComplexData) -> Tuple[frozenset, PHResult]:
    """返回 (与允许开孔同调等价的出生边键集合, 增广复形持久结果)。

    增广只是向复形末尾（按固定同分序）追加面，已有边的过滤序与配对身份不变，
    因此可直接用出生边键在增广结果中定位同一条。
    """
    aug_cd = augmented_complex_data(complex_data)
    if aug_cd is complex_data:
        return frozenset(), ReductionEngine(complex_data).reduce()

    aug_ph = ReductionEngine(aug_cd).reduce()
    opening_face_keys = set(opening_faces_in_window(complex_data))
    allowed_keys: set = set()
    for pair in aug_ph.pairs:
        if pair.dim != 1 or pair.infinite:
            continue
        if aug_cd.cells[pair.death_idx].key in opening_face_keys:
            allowed_keys.add(aug_cd.cells[pair.birth_idx].key)
    return frozenset(allowed_keys), aug_ph


def chain_is_allowed(complex_data: ComplexData, edge_chain: Sequence[int]) -> bool:
    """按代表边链判定其同调类是否属于开孔边界子空间。

    代表链的最大下标边即该类在固定顺序下的出生边；用它在增广复形中查身份。
    """
    allowed_keys, _ = allowed_cycle_births(complex_data)
    if not edge_chain or not allowed_keys:
        return False
    pivot_edge = max(edge_chain)
    return complex_data.cells[pivot_edge].key in allowed_keys


def classify_allowed_cycles(ph_result: PHResult) -> Dict[int, bool]:
    """对结果中的每条 1 维条给出是否与允许开孔同调等价（按出生下标登记）。"""
    cd = ph_result.complex_data
    allowed_keys, _ = allowed_cycle_births(cd)
    verdict: Dict[int, bool] = {}
    for pair in ph_result.pairs:
        if pair.dim != 1:
            continue
        birth_key = cd.cells[pair.birth_idx].key
        verdict[pair.birth_idx] = birth_key in allowed_keys
    return verdict
