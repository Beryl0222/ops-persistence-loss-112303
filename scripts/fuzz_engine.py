"""随机网格模糊测试：分维块引擎 vs 独立的标准全矩阵化简参照实现。"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from topocoat.cells import (
    DIM_E, DIM_F, DIM_V, HORIZONTAL, Cell, edge_vertex_keys,
    face_boundary_keys, filtration_order,
)
from topocoat.engine import run_persistence
from topocoat.observation import ComplexData, ObservationModel, Tile


def reference_persistence(cd):
    """教科书式左到右全边界矩阵化简，返回 (dim, birth, death) 多重集。

    有限配对：主元 p 所在列 j 与 p 配对 (dim(p), p, j)；
    无限条：从未被任何列用作主元的空列（真正的同调出生）。
    """
    n = len(cd.cells)
    columns = [list(b) for b in cd.boundaries]
    pivot_to_col = {}
    for j in range(n):
        col = columns[j]
        while col:
            pivot = col[-1]
            if pivot in pivot_to_col:
                donor = pivot_to_col[pivot]
                a, b = col, columns[donor]
                i = k = 0
                out = []
                while i < len(a) and k < len(b):
                    if a[i] < b[k]:
                        out.append(a[i]); i += 1
                    elif a[i] > b[k]:
                        out.append(b[k]); k += 1
                    else:
                        i += 1; k += 1
                col = out + a[i:] + b[k:]
            else:
                pivot_to_col[pivot] = j
                break
        columns[j] = col
    pairs = set()
    used_as_pivot = set(pivot_to_col)
    for pivot, death_col in pivot_to_col.items():
        pairs.add((cd.cells[pivot].dim, pivot, death_col))
    for j in range(n):
        if j not in used_as_pivot and not columns[j] and cd.cells[j].dim < 2:
            pairs.add((cd.cells[j].dim, j, -1))
    return pairs


def random_model(seed):
    rng = random.Random(seed)
    rows, cols = rng.randint(3, 7), rng.randint(3, 7)
    values = tuple(
        tuple(rng.randint(0, 5) for _ in range(cols))
        for _ in range(rows)
    )
    holes = tuple()
    if rng.random() < 0.5:
        hr, hc = rng.randrange(rows - 1), rng.randrange(cols - 1)
        holes = ((hr, hc),)
    tile = Tile("T", (0, 0), values)
    return ObservationModel((rows, cols), [tile], holes)


def main():
    failures = 0
    for seed in range(300):
        model = random_model(seed)
        cd = model.build_complex()
        ph = run_persistence(cd)
        got = {(p.dim, p.birth_idx, p.death_idx) for p in ph.pairs}
        ref = reference_persistence(cd)
        # 参照实现会包含 2 维空列（平面网格应为零个）；过滤 dim<2 比较。
        ref = {t for t in ref if t[0] < 2}
        if got != ref:
            failures += 1
            print("MISMATCH seed", seed, "missing", ref - got, "extra", got - ref)
            if failures > 3:
                break
    print("fuzz done, failures:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
