"""开孔增广的同调不变量模糊测试。"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from topocoat.engine import run_persistence
from topocoat.observation import ObservationModel, Tile
from topocoat.openings import augmented_complex_data, classify_allowed_cycles


def main():
    rng = random.Random(42)
    failures = 0
    for seed in range(200):
        rows, cols = rng.randint(4, 8), rng.randint(4, 8)
        values = tuple(
            tuple(rng.randint(0, 6) for _ in range(cols)) for _ in range(rows)
        )
        n_holes = rng.randint(0, min(3, (rows - 1) * (cols - 1)))
        candidates = [(r, c) for r in range(rows - 1) for c in range(cols - 1)]
        rng.shuffle(candidates)
        holes = tuple(candidates[:n_holes])
        model = ObservationModel((rows, cols), [Tile("T", (0, 0), values)], holes)
        cd = model.build_complex()
        ph = run_persistence(cd)
        beta1 = len(ph.infinite_bars(1))
        # 期望：平面网格去掉 n 张面 -> β1 恰好为 n（互不相邻时的下界断言用 >= 更稳）。
        if beta1 < n_holes:
            failures += 1
            print("BETA seed", seed, "holes", n_holes, "beta1", beta1)
            continue
        # 增广复形补回全部开孔面后，无限 H1 应清零。
        aug = augmented_complex_data(cd)
        aug_ph = run_persistence(aug)
        if aug_ph.infinite_bars(1):
            failures += 1
            print("AUG seed", seed, "remaining infinite H1", len(aug_ph.infinite_bars(1)))
            continue
        # 允许判定的数量等于无限 H1 条数（所有无限条都由开孔产生）。
        verdict = classify_allowed_cycles(ph)
        allowed = sum(1 for ok in verdict.values() if ok)
        if allowed != beta1:
            failures += 1
            print("ALLOW seed", seed, "allowed", allowed, "beta1", beta1)
    print("opening fuzz done, failures:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
