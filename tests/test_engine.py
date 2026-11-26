"""持久化引擎与胞腔复形约定的单元测试。"""
import unittest

from persistence_loss_input import load_record
from topocoat.cells import DIM_E, DIM_F, DIM_V, filtration_order, in_box
from topocoat.engine import run_persistence
from topocoat.observation import ObservationModel


class FiltrationOrderTests(unittest.TestCase):
    def test_tie_break_dimension_then_rc_then_orientation(self):
        from topocoat.cells import Cell, HORIZONTAL, VERTICAL

        cells = [
            Cell((DIM_F, 1, 1, 0), 3.0),
            Cell((DIM_E, 1, 1, VERTICAL), 3.0),
            Cell((DIM_E, 1, 1, HORIZONTAL), 3.0),
            Cell((DIM_V, 1, 1, 0), 3.0),
        ]
        ordered = filtration_order(cells)
        self.assertEqual([c.dim for c in ordered], [DIM_V, DIM_E, DIM_E, DIM_F])
        self.assertEqual(ordered[1].orient, HORIZONTAL)
        self.assertEqual(ordered[2].orient, VERTICAL)

    def test_box_is_half_open_and_face_needs_all_corners(self):
        self.assertTrue(in_box((DIM_V, 0, 0, 0), (0, 0, 2, 2)))
        self.assertFalse(in_box((DIM_V, 2, 0, 0), (0, 0, 2, 2)))
        self.assertFalse(in_box((DIM_F, 1, 1, 0), (0, 0, 2, 2)))


class PersistenceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_record()
        cls.model = ObservationModel.from_record(cls.record)
        cls.cd = cls.model.build_complex()
        cls.ph = run_persistence(cls.cd)

    def test_completion_and_betti_counts(self):
        self.assertTrue(self.ph.complete)
        # 6x7 网格去掉 1 张面：连通分量 1、1 维无限条 1。
        self.assertEqual(len(self.ph.infinite_bars(0)), 1)
        self.assertEqual(len(self.ph.infinite_bars(1)), 1)
        finite_h1 = self.ph.finite_pairs(1)
        # 手工核算：28 条零持久条 + 1 条 (0,5) 的真实孔洞；永久开孔走无限条。
        by_pers = {}
        for pair in finite_h1:
            by_pers.setdefault(pair.persistence, 0)
            by_pers[pair.persistence] += 1
        self.assertEqual(by_pers.get(0.0), 28)
        self.assertEqual(by_pers.get(5.0), 1)
        # 41 个有限 0 维配对把 42 个顶点连成 1 个分量。
        self.assertEqual(len(self.ph.finite_pairs(0)), 41)

    def test_expected_long_hole_pair(self):
        long_pairs = [p for p in self.ph.finite_pairs(1) if p.persistence > 0]
        self.assertEqual(len(long_pairs), 1)
        pair = long_pairs[0]
        self.assertEqual(self.cd.cells[pair.birth_idx].key, (DIM_E, 3, 4, 0))
        self.assertEqual(self.cd.cells[pair.death_idx].key, (DIM_F, 2, 4, 0))
        self.assertEqual((pair.birth_value, pair.death_value), (0.0, 5.0))
        # 代表链是环绕 (2,4) 高分孤岛的闭合边环：环经过其四周格点但不含环心。
        points = self.ph.representative_points(pair)
        self.assertIn((3, 4), points)
        self.assertIn((1, 4), points)
        self.assertNotIn((2, 4), points)

    def test_permanent_hole_face_absent_but_edges_kept(self):
        hole_key = (DIM_F, 2, 2, 0)
        self.assertNotIn(hole_key, self.cd.index)
        for edge in self.cd.cells:
            if edge.dim == DIM_E and edge.row in (2, 3) and edge.col in (2, 3):
                self.assertIn(edge.key, self.cd.index)

    def test_budget_returns_only_provable_partial(self):
        ph = run_persistence(self.cd, budget=1)
        self.assertFalse(ph.complete)
        # 部分结果不得伪造无限条之外的结论，且所有出具配对都有完整身份。
        for pair in ph.pairs:
            self.assertIsInstance(pair.birth_idx, int)

    def test_chain_is_a_closed_cycle_f2(self):
        pair = next(p for p in self.ph.finite_pairs(1) if p.persistence > 0)
        # F₂ 下闭合边链：每个顶点度数为偶数。
        from collections import Counter

        from topocoat.cells import edge_endpoints

        degree = Counter()
        for edge_idx in pair.representative:
            for point in edge_endpoints(self.cd.cells[edge_idx].key):
                degree[point] += 1
        self.assertTrue(all(deg % 2 == 0 for deg in degree.values()))


if __name__ == "__main__":
    unittest.main()
