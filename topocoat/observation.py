"""分片观测模型：声明等价合并、缺片与永久开孔区分、复形构建与观测前沿。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .cells import (
    DIM_E,
    DIM_F,
    DIM_V,
    HORIZONTAL,
    VERTICAL,
    Cell,
    CellKey,
    edge_vertex_keys,
    face_boundary_keys,
    face_corners,
    filtration_order,
)

Point = Tuple[int, int]
Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Tile:
    """按工艺范围打分的矩形片；``origin`` 是左上角格点索引，``values`` 按行、列。"""

    id: str
    origin: Point
    values: Tuple[Tuple[float, ...], ...]

    @property
    def rows(self) -> int:
        return len(self.values)

    @property
    def cols(self) -> int:
        return len(self.values[0]) if self.values else 0

    def covers(self, r: int, c: int) -> bool:
        r0, c0 = self.origin
        return r0 <= r < r0 + self.rows and c0 <= c < c0 + self.cols

    def score(self, r: int, c: int) -> float:
        r0, c0 = self.origin
        return float(self.values[r - r0][c - c0])

    @classmethod
    def from_record_item(cls, item: Mapping) -> "Tile":
        return cls(
            id=str(item["id"]),
            origin=(int(item["origin"][0]), int(item["origin"][1])),
            values=tuple(tuple(float(v) for v in row) for row in item["values"]),
        )


@dataclass(frozen=True)
class TileConflict:
    """同一格点被多个分片以不同分数声明。"""

    point: Point
    claims: Tuple[Tuple[str, float], ...]


class TileConflictError(ValueError):
    def __init__(self, conflicts: Sequence[TileConflict]):
        self.conflicts = tuple(conflicts)
        super().__init__(
            "重叠格点声明不一致："
            + "；".join(f"{p} -> {c}" for p, c in ((x.point, x.claims) for x in conflicts))
        )


@dataclass(frozen=True)
class ComplexData:
    """一次复形构建的全部输入事实，引擎直接消费。"""

    cells: Tuple[Cell, ...]                       # 固定过滤顺序
    index: Dict[CellKey, int]
    boundaries: Tuple[Tuple[int, ...], ...]       # 每个单元的边界（下标，F₂）
    frontier: frozenset[int]                      # 与缺片区域相邻、结论可被后续数据改变的单元
    unknown_vertices: frozenset[Point]            # 还没送到的格点
    provenance: Tuple[Tuple[str, ...], ...]       # 每个单元的计分身份（分片 id，按声明顺序）
    shape: Tuple[int, int]
    box: Box
    permanent_holes: frozenset[Point]

    def cell(self, idx: int) -> Cell:
        return self.cells[idx]

    def key_of(self, idx: int) -> CellKey:
        return self.cells[idx].key

    def is_frontier(self, idx: int) -> bool:
        return idx in self.frontier

    def fully_observed(self) -> bool:
        return not self.unknown_vertices


class ObservationModel:
    """有归属的表面网格：持有分片、永久开孔与合并后的规范格点分数。

    重叠格点只有在所有声明等价（分数相同）时才能合并；规范变量身份始终是全局
    (行, 列)，多片重复声明不会变成多个变量。
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        tiles: Sequence[Tile],
        permanent_face_holes: Sequence[Point] = (),
        delivered: Optional[Sequence[str]] = None,
    ):
        rows, cols = shape
        if rows <= 0 or cols <= 0:
            raise ValueError("曲面尺寸必须为正")
        self.shape = (int(rows), int(cols))
        self.tiles: Tuple[Tile, ...] = tuple(tiles)
        self._tile_order: Dict[str, int] = {tile.id: i for i, tile in enumerate(self.tiles)}
        if len(self._tile_order) != len(self.tiles):
            raise ValueError("分片 id 必须唯一")
        self.permanent_holes: frozenset[Point] = frozenset((int(r), int(c)) for r, c in permanent_face_holes)
        for r, c in self.permanent_holes:
            if not (0 <= r < rows - 1 and 0 <= c < cols - 1):
                raise ValueError(f"永久开孔面 {r,c} 超出曲面范围")
        delivered_ids = set(delivered) if delivered is not None else {tile.id for tile in self.tiles}
        unknown_tile = delivered_ids.difference(self._tile_order)
        if unknown_tile:
            raise ValueError(f"未知分片 id：{sorted(unknown_tile)}")
        self.delivered: frozenset[str] = frozenset(delivered_ids)
        self._claims: Dict[Point, List[Tuple[str, float]]] = {}
        self._scores: Dict[Point, float] = {}
        conflicts: List[TileConflict] = []
        for tile in self.tiles:
            if tile.id not in self.delivered:
                continue
            self._check_tile_extent(tile)
            for r in range(tile.rows):
                for c in range(tile.cols):
                    point = (tile.origin[0] + r, tile.origin[1] + c)
                    value = tile.score(*point)
                    self._claims.setdefault(point, []).append((tile.id, value))
        for point, claims in self._claims.items():
            values = {value for _, value in claims}
            if len(values) > 1:
                conflicts.append(TileConflict(point=point, claims=tuple(claims)))
        if conflicts:
            raise TileConflictError(conflicts)
        for point, claims in self._claims.items():
            self._scores[point] = claims[0][1]

    def _check_tile_extent(self, tile: Tile) -> None:
        rows, cols = self.shape
        r0, c0 = tile.origin
        if not (0 <= r0 and 0 <= c0 and r0 + tile.rows <= rows and c0 + tile.cols <= cols):
            raise ValueError(f"分片 {tile.id} 超出曲面范围")
        if any(len(row) != tile.cols for row in tile.values):
            raise ValueError(f"分片 {tile.id} 必须是矩形片")

    # ---- 基本查询 -------------------------------------------------------
    @classmethod
    def from_record(cls, record: Mapping, delivered: Optional[Sequence[str]] = None) -> "ObservationModel":
        return cls(
            shape=(int(record["surface"][0]), int(record["surface"][1])),
            tiles=[Tile.from_record_item(item) for item in record["tiles"]],
            permanent_face_holes=[tuple(p) for p in record.get("permanent_face_holes", ())],
            delivered=delivered,
        )

    def is_observed(self, r: int, c: int) -> bool:
        return (r, c) in self._scores

    def score(self, r: int, c: int) -> Optional[float]:
        """规范格点分数；缺片未送到时返回 ``None``，与永久开孔无关。"""
        return self._scores.get((r, c))

    def point_tile_ids(self, r: int, c: int) -> Tuple[str, ...]:
        """该格点的计分身份：声明过它的分片，按分片声明顺序。"""
        claims = self._claims.get((r, c), ())
        return tuple(tile_id for tile_id, _ in sorted(claims, key=lambda x: self._tile_order[x[0]]))

    def all_unknown_vertices(self) -> frozenset[Point]:
        rows, cols = self.shape
        return frozenset((r, c) for r in range(rows) for c in range(cols) if not self.is_observed(r, c))

    def fingerprint(self) -> str:
        """当前观测状态的规范指纹：复核台账与训练复用同一状态时据此核对。"""
        payload = {
            "shape": list(self.shape),
            "holes": sorted(list(self.permanent_holes)),
            "delivered": sorted(self.delivered),
            "claims": [
                [list(point), [[tid, val] for tid, val in sorted(claims, key=lambda x: self._tile_order[x[0]])]]
                for point, claims in sorted(self._claims.items())
            ],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    # ---- 复形构建 -------------------------------------------------------
    def build_complex(self, box: Optional[Box] = None, halo: int = 0) -> ComplexData:
        return build_complex(self, box=box, halo=halo)


def _expand_box(shape: Tuple[int, int], box: Optional[Box], halo: int) -> Box:
    rows, cols = shape
    if box is None:
        base = (0, 0, rows, cols)
    else:
        r0, c0, r1, c1 = box
        if not (0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols):
            raise ValueError("box 必须是曲面内的半开顶点区间")
        base = (r0, c0, r1, c1)
    r0, c0, r1, c1 = base
    return (
        max(0, r0 - halo),
        max(0, c0 - halo),
        min(rows, r1 + halo),
        min(cols, c1 + halo),
    )


def build_complex(model: ObservationModel, box: Optional[Box] = None, halo: int = 0) -> ComplexData:
    """构建（可裁取的）胞腔复形。

    * 顶点：box（含 halo）内已观测格点；
    * 边：两个端点都已观测且在 box 内；
    * 面：四个角点都已观测、在 box 内，且不属于永久开孔；
    * 缺片格点形成的是 *观测前沿*，不生成任何单元；永久开孔只去掉面，
      其四条边和四个顶点照常生效——两者语义不混淆。
    """
    if halo < 0:
        raise ValueError("halo 不能为负")
    eff_box = _expand_box(model.shape, box, halo)
    r0, c0, r1, c1 = eff_box
    unknown = frozenset(
        (r, c)
        for r in range(r0, r1)
        for c in range(c0, c1)
        if not model.is_observed(r, c)
    )

    cells: List[Cell] = []
    keys: List[CellKey] = []
    provenance: List[Tuple[str, ...]] = []

    def add_cell(key: CellKey, value: float, tile_ids: Set[str]) -> None:
        cells.append(Cell(key=key, value=float(value)))
        keys.append(key)
        provenance.append(tuple(sorted(tile_ids, key=lambda tid: model._tile_order[tid])))

    # 顶点
    for r in range(r0, r1):
        for c in range(c0, c1):
            if model.is_observed(r, c):
                add_cell((DIM_V, r, c, 0), model.score(r, c), set(model.point_tile_ids(r, c)))
    # 边（横、竖）
    for r in range(r0, r1):
        for c in range(c0, c1):
            if c + 1 < c1 and model.is_observed(r, c) and model.is_observed(r, c + 1):
                key = (DIM_E, r, c, HORIZONTAL)
                tids = set(model.point_tile_ids(r, c)) | set(model.point_tile_ids(r, c + 1))
                add_cell(key, max(model.score(r, c), model.score(r, c + 1)), tids)
            if r + 1 < r1 and model.is_observed(r, c) and model.is_observed(r + 1, c):
                key = (DIM_E, r, c, VERTICAL)
                tids = set(model.point_tile_ids(r, c)) | set(model.point_tile_ids(r + 1, c))
                add_cell(key, max(model.score(r, c), model.score(r + 1, c)), tids)
    # 面（永久开孔面不进入；边和顶点保留）
    for r in range(r0, r1 - 1):
        for c in range(c0, c1 - 1):
            key = (DIM_F, r, c, 0)
            if (r, c) in model.permanent_holes:
                continue
            corners = face_corners(key)
            if all(model.is_observed(rr, cc) for rr, cc in corners):
                value = max(model.score(rr, cc) for rr, cc in corners)
                tids: Set[str] = set()
                for rr, cc in corners:
                    tids.update(model.point_tile_ids(rr, cc))
                add_cell(key, value, tids)

    ordered = filtration_order(cells)
    permutation = [keys.index(cell.key) for cell in ordered]
    index = {cell.key: i for i, cell in enumerate(ordered)}
    ordered_provenance = tuple(provenance[p] for p in permutation)

    boundaries: List[Tuple[int, ...]] = []
    for cell in ordered:
        if cell.dim == DIM_V:
            boundaries.append(())
        elif cell.dim == DIM_E:
            boundaries.append(tuple(sorted(index[k] for k in edge_vertex_keys(cell.key))))
        else:
            boundaries.append(tuple(sorted(index[k] for k in face_boundary_keys(cell.key) if k in index)))

    frontier = frontier_cells(ordered, index, unknown, model.permanent_holes)
    return ComplexData(
        cells=tuple(ordered),
        index=index,
        boundaries=tuple(boundaries),
        frontier=frontier,
        unknown_vertices=unknown,
        provenance=ordered_provenance,
        shape=model.shape,
        box=eff_box,
        permanent_holes=model.permanent_holes,
    )


def build_window_complex(model: ObservationModel, box: Box, halo: int = 0) -> ComplexData:
    """不改变复核范围边界语义的裁取：box 外单元不进入，halo 仅用于可证明性。"""
    return build_complex(model, box=box, halo=halo)


def frontier_cells(
    ordered_cells: Sequence[Cell],
    index: Mapping[CellKey, int],
    unknown_vertices: frozenset[Point],
    permanent_holes: frozenset[Point],
) -> frozenset[int]:
    """观测前沿：已建单元中，边界顶点与缺片格点相邻者。

    缺片只可能在与未知格点相邻的位置补入新单元，因此内部单元（所有顶点的
    闭星都已观测）参与的配对不受后续数据影响。永久开孔是固定几何，不产生前沿。
    """
    if not unknown_vertices:
        return frozenset()
    unknown_set = set(unknown_vertices)
    frontier_keys: Set[CellKey] = set()
    for cell in ordered_cells:
        key = cell.key
        if cell.dim == DIM_V:
            r, c = key[1], key[2]
            incident_points = [(r, c), *_grid_neighbors(r, c)]
        elif cell.dim == DIM_E:
            va, vb = edge_vertex_keys(key)
            incident_points = [(va[1], va[2]), (vb[1], vb[2])]
        else:
            incident_points = list(face_corners(key))
        if any(p in unknown_set for p in incident_points):
            frontier_keys.add(key)
    return frozenset(index[k] for k in frontier_keys if k in index)


def _grid_neighbors(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
