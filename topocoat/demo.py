"""端到端演示：同一份分块打分贯通复核、放行、台账替代与训练信号。

用法：
    python -m topocoat.demo                 # 使用内置交换样本
    python -m topocoat.demo path/to.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from .observation import ObservationModel
from .review import RELEASE, TICKET, ReviewLedger, ReviewOrder, review_workpiece
from .service import TopoService
from .training import TrainingRequest, persistence_loss, resolve_proposal


def _load(path: str | None):
    # 允许在未安装为包时从仓库根读取交换样本入口。
    try:
        from persistence_loss_input import load_record as _load_record
    except ImportError:  # pragma: no cover
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root))
        from persistence_loss_input import load_record as _load_record
    return _load_record(path) if path else _load_record()


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    record = _load(argv[0] if argv else None)
    orders: Dict[str, ReviewOrder] = {
        rid: ReviewOrder.from_record(rid, spec) for rid, spec in record["regions"].items()
    }

    print("== 1. 缺片到达前（仅 L 片）：只给可证明的部分 ==")
    partial = ObservationModel.from_record(record, delivered=["L"])
    ledger = ReviewLedger()
    partial_entries, _ = review_workpiece(partial, orders)
    ledger.update(partial_entries)
    for entry in partial_entries:
        print(f"  [{entry.verdict:7}] {entry.pair_identity[0]} | {entry.note}")

    print("\n== 2. 全工件到达：旧未定被可证明结论替代 ==")
    full = ObservationModel.from_record(record)
    events = ledger.update(review_workpiece(full, orders)[0])
    for event in events:
        print(" ", event)
    for entry in ledger.entries:
        if entry.verdict in (TICKET, RELEASE):
            print(f"  [{entry.verdict:7}] 分区{entry.region_ids} 身份{entry.pair_identity} "
                  f"持久度={entry.persistence} 分片={entry.tile_ids}")
            if entry.verdict == TICKET:
                print(f"            代表路径经过: {entry.path_regions} {entry.representative_path}")

    print("\n== 3. 工艺员提案解析（消失格点 (2,4) 有多个候选，须人工选择） ==")
    service = TopoService(full)
    ph = service.full_persistence()
    candidates = resolve_proposal(ph, (2, 4))
    for i, pair in enumerate(candidates):
        print(f"  候选{i}: 身份={pair.identity(ph.complex_data)} 持久度={pair.persistence}")
    chosen = candidates[0]

    print("\n== 4. 训练损失与原始格点梯度（规范变量，重叠格点不重复计） ==")
    req = TrainingRequest.make(chosen.identity(ph.complex_data), target_delta=+1.0, weight=0.5)
    outcome = persistence_loss(ph, [req], strict_ties=False)
    print(f"  损失={outcome.total_loss:.4f}")
    for point, grad in sorted(outcome.gradient_by_point.items()):
        print(f"  格点{point} 梯度={grad:+.4f}")
    strict = persistence_loss(ph, [req], strict_ties=True)
    print("  严格同分模式:", strict.outcomes[0].status, strict.outcomes[0].skip_reason)

    print("\n== 5. 条带化复核与全量结果一致（正持久孔洞身份逐对相同） ==")
    stripes = service.stripe_review(stripe_rows=3)
    positive = set()
    for stripe in stripes:
        for pair in stripe.certified_pairs:
            if pair.dim == 1 and not pair.infinite and pair.persistence > 0:
                positive.add((stripe.ph.complex_data.cells[pair.birth_idx].key,
                              stripe.ph.complex_data.cells[pair.death_idx].key))
    print("  条带认证的正持久孔洞:", positive)

    print("\n== 6. 孔几何对比：同分数、不同人工孔几何 -> 无限 H1 数 0/1/2 ==")
    for name, holes in (("无开孔", ()), ("单开孔", ((2, 2),)), ("双开孔", ((2, 2), (1, 4)))):
        model = ObservationModel(full.shape, full.tiles, permanent_face_holes=holes)
        from .engine import run_persistence

        count = len(run_persistence(model.build_complex()).infinite_bars(1))
        print(f"  {name}: 无限 1 维条 {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
