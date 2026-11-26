"""分块曲面的持久拓扑筛查与学习算子。

约定（与 ``data/record.json`` 交换样本配套）：

* 方形格点复形，坐标按 (行, 列)；
* 分区 box 是半开顶点索引区间 [r0, c0, r1, c1)，边和面仅当全部顶点在域内有效；
* 格点分数定义子水平过滤，边/面在全部顶点进入后生效（过滤值取顶点分数最大值）；
* 同分时按维度升序、同维按行列、后接方向（横、竖）排序，使面在边后；
* 系数域 F₂；
* ``permanent_face_holes`` 给出永久去掉的方格面左上顶点：面不进入复形，
  四条边和四个顶点保留。
"""

from .cells import (
    DIM_E,
    DIM_F,
    DIM_V,
    HORIZONTAL,
    VERTICAL,
    Cell,
    CellKey,
    edge_endpoints,
    face_boundary_keys,
    face_corners,
    neighbor_vertices,
)
from .engine import PHResult, PersistencePair, ReductionEngine, run_persistence
from .observation import (
    ObservationModel,
    Tile,
    TileConflict,
    TileConflictError,
    build_complex,
    build_window_complex,
    frontier_cells,
)
from .openings import augmented_complex_data, chain_is_allowed, classify_allowed_cycles
from .review import (
    PENDING,
    RELEASE,
    TICKET,
    BudgetExhausted,
    ReviewLedger,
    ReviewOrder,
    review_region,
    review_workpiece,
)
from .service import StripeResult, TopoService
from .training import (
    SKIP_IDENTITY,
    SKIP_TIE,
    SKIP_UNDETERMINED,
    TrainingOutcome,
    TrainingRequest,
    persistence_loss,
    resolve_proposal,
)

__all__ = [
    "DIM_E",
    "DIM_F",
    "DIM_V",
    "HORIZONTAL",
    "VERTICAL",
    "Cell",
    "CellKey",
    "edge_endpoints",
    "face_boundary_keys",
    "face_corners",
    "neighbor_vertices",
    "PHResult",
    "PersistencePair",
    "ReductionEngine",
    "run_persistence",
    "ObservationModel",
    "Tile",
    "TileConflict",
    "TileConflictError",
    "build_complex",
    "build_window_complex",
    "frontier_cells",
    "augmented_complex_data",
    "chain_is_allowed",
    "classify_allowed_cycles",
    "PENDING",
    "RELEASE",
    "TICKET",
    "BudgetExhausted",
    "ReviewLedger",
    "ReviewOrder",
    "review_region",
    "review_workpiece",
    "StripeResult",
    "TopoService",
    "SKIP_IDENTITY",
    "SKIP_TIE",
    "SKIP_UNDETERMINED",
    "TrainingOutcome",
    "TrainingRequest",
    "persistence_loss",
    "resolve_proposal",
]
