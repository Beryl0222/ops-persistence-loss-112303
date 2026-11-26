"""方形胞腔复形的单元身份与固定过滤顺序。

单元身份是跨分片、跨换图的稳定引用（训练侧据此定位原始格点）：

* 顶点 ``(DIM_V, r, c, 0)``
* 横边 ``(DIM_E, r, c, 0)``：连接 (r,c)-(r,c+1)
* 竖边 ``(DIM_E, r, c, 1)``：连接 (r,c)-(r+1,c)
* 方格面 ``(DIM_F, r, c, 0)``：左上顶点 (r,c)

同分排序：维度升序；同维按行、列；再按方向（横 0、竖 1）。
因此同分上面始终在其四条边之后，边在顶点之后，边界单元严格先于单元本身。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

#: 维度枚举：顶点 0、边 1、面 2。
DIM_V = 0
DIM_E = 1
DIM_F = 2

#: 方向枚举（仅边使用）：横、竖。
HORIZONTAL = 0
VERTICAL = 1

#: 单元键 ``(维度, 行, 列, 方向/占位)``。
CellKey = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Cell:
    """复形中的一个单元；``value`` 是其过滤值（顶点分数或顶点分数最大值）。"""

    key: CellKey
    value: float

    @property
    def dim(self) -> int:
        return self.key[0]

    @property
    def row(self) -> int:
        return self.key[1]

    @property
    def col(self) -> int:
        return self.key[2]

    @property
    def orient(self) -> int:
        return self.key[3]


def edge_endpoints(key: CellKey) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """返回边键两个端点的 (行, 列)。"""
    _, r, c, orient = key
    if orient == HORIZONTAL:
        return (r, c), (r, c + 1)
    return (r, c), (r + 1, c)


def face_corners(key: CellKey) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """返回面键的四个角点（左上、右上、左下、右下）。"""
    _, r, c, _ = key
    return (r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)


def face_boundary_keys(key: CellKey) -> Tuple[CellKey, CellKey, CellKey, CellKey]:
    """面的四条边界边（上、左、下、右），顺序本身无语义，F₂ 下无方向。"""
    _, r, c, _ = key
    return (
        (DIM_E, r, c, HORIZONTAL),      # 上边
        (DIM_E, r, c, VERTICAL),        # 左边
        (DIM_E, r + 1, c, HORIZONTAL),  # 下边
        (DIM_E, r, c + 1, VERTICAL),    # 右边
    )


def edge_vertex_keys(key: CellKey) -> Tuple[CellKey, CellKey]:
    """边的两个顶点键。"""
    (r1, c1), (r2, c2) = edge_endpoints(key)
    return (DIM_V, r1, c1, 0), (DIM_V, r2, c2, 0)


def neighbor_vertices(r: int, c: int, rows: int, cols: int) -> Iterable[Tuple[int, int]]:
    """网格上 (r,c) 的四邻格点（在曲面范围内）。"""
    if r > 0:
        yield r - 1, c
    if r + 1 < rows:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < cols:
        yield r, c + 1


def filtration_order(cells: Iterable[Cell]) -> List[Cell]:
    """固定过滤顺序：先过滤值，再维度，再行、列、方向。

    这是唯一合法的同分裁决顺序；训练侧的同分风险判定也围绕它展开。
    """
    return sorted(cells, key=lambda cell: (cell.value, cell.dim, cell.row, cell.col, cell.orient))


def in_box(key: CellKey, box: Sequence[int]) -> bool:
    """单元的所有顶点是否落在半开 box [r0,c0,r1,c1) 内。"""
    r0, c0, r1, c1 = box
    dim = key[0]
    if dim == DIM_V:
        r, c = key[1], key[2]
        return r0 <= r < r1 and c0 <= c < c1
    if dim == DIM_E:
        (a_r, a_c), (b_r, b_c) = edge_endpoints(key)
        return all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in ((a_r, a_c), (b_r, b_c)))
    corners = face_corners(key)
    return all(r0 <= rr < r1 and c0 <= cc < c1 for rr, cc in corners)
