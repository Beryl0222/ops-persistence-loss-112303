"""允许开孔同调等价判定的测试。"""
import unittest

from persistence_loss_input import load_record
from topocoat import ObservationModel, augmented_complex_data, classify_allowed_cycles, run_persistence


class OpeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_record()
        cls.model = ObservationModel.from_record(cls.record)
        cls.cd = cls.model.build_complex()
        cls.ph = run_persistence(cls.cd)
        cls.verdict = classify_allowed_cycles(cls.ph)

    def test_permanent_hole_loop_is_allowed(self):
        infinite = self.ph.infinite_bars(1)
        self.assertEqual(len(infinite), 1)
        birth = infinite[0].birth_idx
        self.assertTrue(self.verdict[birth])

    def test_real_defect_loop_is_not_allowed(self):
        long_pair = next(p for p in self.ph.finite_pairs(1) if p.persistence > 0)
        self.assertFalse(self.verdict[long_pair.birth_idx])

    def test_augmented_complex_kills_exactly_opening_class(self):
        aug_cd = augmented_complex_data(self.cd)
        # 补回一张开孔面：面数 +1，边和顶点数不变。
        self.assertEqual(
            sum(1 for c in aug_cd.cells if c.dim == 2),
            sum(1 for c in self.cd.cells if c.dim == 2) + 1,
        )
        self.assertEqual(
            sum(1 for c in aug_cd.cells if c.dim == 1),
            sum(1 for c in self.cd.cells if c.dim == 1),
        )
        aug_ph = run_persistence(aug_cd)
        # 增广后 1 维无限条清零（开孔面闭合了唯一的允许环）。
        self.assertEqual(len(aug_ph.infinite_bars(1)), 0)
        # 缺陷环仍然在普通面 (2,4) 处闭合，身份保持。
        death_keys = {
            aug_cd.cells[p.death_idx].key
            for p in aug_ph.finite_pairs(1) if p.persistence > 0
        }
        self.assertIn((2, 2, 4, 0), death_keys)

    def test_two_openings_give_two_allowed_classes(self):
        two = ObservationModel(
            self.model.shape, self.model.tiles,
            permanent_face_holes=((2, 2), (1, 4)),
        )
        ph = run_persistence(two.build_complex())
        verdict = classify_allowed_cycles(ph)
        allowed = [idx for idx, ok in verdict.items() if ok]
        self.assertEqual(len(allowed), 2)


if __name__ == "__main__":
    unittest.main()
