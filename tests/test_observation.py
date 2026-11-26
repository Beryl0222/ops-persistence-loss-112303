"""分片观测、缺片语义、前沿与同分工单冲突的测试。"""
import unittest

from persistence_loss_input import load_record
from topocoat import (
    DIM_E,
    DIM_V,
    ObservationModel,
    Tile,
    TileConflictError,
    run_persistence,
)
from topocoat.cells import edge_endpoints


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_record()

    def test_overlap_equivalence_merges_to_one_variable(self):
        model = ObservationModel.from_record(self.record)
        # (r,3) 是两片的重叠列：两个分片声明等价，规范身份只有一个变量。
        self.assertEqual(model.score(2, 3), 0.0)
        ids = model.point_tile_ids(2, 3)
        self.assertEqual(ids, ("L", "R"))

    def test_conflicting_overlap_raises(self):
        # 两个同域分片在 (2,4) 给出不同分数 -> 重叠声明不等价，拒绝合并。
        from copy import deepcopy

        item_r = deepcopy(self.record["tiles"][1])
        item_r2 = deepcopy(self.record["tiles"][1])
        item_r2["id"] = "R2"
        item_r2["values"][2][1] += 7
        tiles = [
            Tile.from_record_item(self.record["tiles"][0]),
            Tile.from_record_item(item_r),
            Tile.from_record_item(item_r2),
        ]
        with self.assertRaises(TileConflictError) as ctx:
            ObservationModel(self.record["surface"], tiles,
                             self.record["permanent_face_holes"])
        self.assertEqual(ctx.exception.conflicts[0].point, (2, 4))

    def test_missing_tile_creates_frontier_but_not_fake_holes(self):
        partial = ObservationModel.from_record(self.record, delivered=["L"])
        cd = partial.build_complex()
        # 右片 18 个格点未观测；前沿只覆盖与缺片相邻的已建单元。
        self.assertEqual(len(cd.unknown_vertices), 18)
        self.assertGreater(len(cd.frontier), 0)
        ph = run_persistence(cd)
        # 所有 1 维无限条都必须未证明：不能把缺片临时边界当成孔洞。
        for pair in ph.infinite_bars(1):
            self.assertFalse(pair.certified)
            self.assertIsNotNone(pair.uncertified_reason)

    def test_missing_vertex_vs_permanent_opening_distinguished(self):
        partial = ObservationModel.from_record(self.record, delivered=["L"])
        # 永久开孔是固定几何：缺片时其边仍在；缺片格点不产生任何单元。
        cd = partial.build_complex()
        hole_edges = [
            cell for cell in cd.cells
            if cell.dim == DIM_E and cell.row in (2, 3) and cell.col in (2, 3)
        ]
        self.assertTrue(hole_edges)
        # R 片区域顶点完全不存在于复形。
        self.assertNotIn((DIM_V, 1, 5, 0), cd.index)

    def test_crop_preserves_boundary_semantics(self):
        model = ObservationModel.from_record(self.record)
        full = model.build_complex()
        crop = model.build_complex(box=(0, 3, 6, 7))
        # 裁取后任何单元的顶点都不得越过 box。
        from topocoat.cells import in_box

        for cell in crop.cells:
            self.assertTrue(in_box(cell.key, crop.box))
        # 裁取复形是全复形的子集（同键、同过滤值）。
        for cell in crop.cells:
            self.assertEqual(full.cells[full.index[cell.key]].value, cell.value)

    def test_fingerprint_changes_with_arrival(self):
        partial = ObservationModel.from_record(self.record, delivered=["L"])
        full = ObservationModel.from_record(self.record)
        self.assertNotEqual(partial.fingerprint(), full.fingerprint())


if __name__ == "__main__":
    unittest.main()
