"""条带化等价性与换图兼容性的测试。"""
import unittest

from persistence_loss_input import load_record
from topocoat import ObservationModel, Tile, TopoService, run_persistence


class StripeAndCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_record()
        cls.model = ObservationModel.from_record(cls.record)
        cls.service = TopoService(cls.model)
        cls.full_ph = cls.service.full_persistence()

    def _full_h1_keys(self):
        return {
            (self.full_ph.complex_data.cells[p.birth_idx].key,
             self.full_ph.complex_data.cells[p.death_idx].key): p.persistence
            for p in self.full_ph.finite_pairs(1)
        }

    def test_stripe_certified_pairs_match_full_persistence(self):
        full_keys = self._full_h1_keys()
        for stripe_rows in (2, 3, 4):
            stripes = self.service.stripe_review(stripe_rows)
            seen = {}
            for stripe in stripes:
                for pair in stripe.certified_pairs:
                    if pair.dim != 1 or pair.infinite:
                        continue
                    key = (
                        stripe.ph.complex_data.cells[pair.birth_idx].key,
                        stripe.ph.complex_data.cells[pair.death_idx].key,
                    )
                    seen.setdefault(key, pair.persistence)
            # 正持久条（真实孔洞）身份必须与全量逐对相同。
            positive_full = {k for k, v in full_keys.items() if v > 0}
            positive_stripe = {k for k, v in seen.items() if v > 0}
            self.assertEqual(positive_stripe, positive_full)
            # 所有正持久全量身份都被条带认证覆盖。
            self.assertTrue(positive_full <= set(seen))

    def test_infinite_bar_only_certified_in_sealed_last_stripe(self):
        stripes = self.service.stripe_review(3)
        for stripe in stripes[:-1]:
            self.assertFalse(
                any(p.infinite and p.certified for p in stripe.certified_pairs)
            )
        last = stripes[-1]
        inf = {
            last.ph.complex_data.cells[p.birth_idx].key
            for p in last.certified_pairs if p.infinite and p.dim == 1
        }
        full_inf = {
            self.full_ph.complex_data.cells[p.birth_idx].key
            for p in self.full_ph.infinite_bars(1)
        }
        self.assertEqual(inf, full_inf)

    def test_compatible_graph_review_identical(self):
        clone = ObservationModel.from_record(self.record)
        TopoService.check_compatible(self.model, clone)
        ph_clone = run_persistence(clone.build_complex())
        self.assertEqual(
            [(p.birth_idx, p.death_idx, p.persistence) for p in ph_clone.pairs],
            [(p.birth_idx, p.death_idx, p.persistence) for p in self.full_ph.pairs],
        )

    def test_incompatible_graph_rejected(self):
        changed = [
            Tile(t.id, t.origin, tuple(
                tuple(v + 1 if (r, c) == (1, 1) else v for c, v in enumerate(row))
                for r, row in enumerate(t.values)
            ))
            for t in self.model.tiles
        ]
        other = ObservationModel(self.model.shape, changed, self.model.permanent_holes)
        with self.assertRaises(ValueError):
            TopoService.check_compatible(self.model, other)

    def test_different_hole_geometry_shows_real_difference(self):
        no_hole = ObservationModel(self.model.shape, self.model.tiles, ())
        two_holes = ObservationModel(
            self.model.shape, self.model.tiles, ((2, 2), (1, 4))
        )
        counts = []
        for model in (no_hole, self.model, two_holes):
            ph = run_persistence(model.build_complex())
            counts.append(len(ph.infinite_bars(1)))
        self.assertEqual(counts, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
