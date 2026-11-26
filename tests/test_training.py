"""训练损失、规范格点梯度、去重与跳过原因的测试。"""
import unittest

from persistence_loss_input import load_record
from topocoat import (
    SKIP_IDENTITY,
    SKIP_TIE,
    SKIP_UNDETERMINED,
    DIM_E,
    DIM_F,
    ObservationModel,
    TrainingRequest,
    TopoService,
    run_persistence,
)
from topocoat.training import resolve_proposal


def bump(model, r, c, delta):
    from topocoat.observation import Tile

    tiles = []
    for tile in model.tiles:
        if tile.id not in model.delivered:
            tiles.append(tile)
            continue
        values = [list(row) for row in tile.values]
        if tile.covers(r, c):
            values[r - tile.origin[0]][c - tile.origin[1]] += delta
        tiles.append(Tile(tile.id, tile.origin, tuple(tuple(row) for row in values)))
    return ObservationModel(model.shape, tiles, model.permanent_holes, model.delivered)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_record()
        cls.model = ObservationModel.from_record(cls.record)
        cls.service = TopoService(cls.model)
        cls.ph = cls.service.full_persistence()
        cls.candidates = resolve_proposal(cls.ph, (2, 4))
        cls.chosen = cls.candidates[0]
        cls.identity = cls.chosen.identity(cls.ph.complex_data)

    def test_proposal_returns_all_candidates_for_technician_choice(self):
        # 提案点 (2,4) 对应 1 个正持久配对和 3 个零持久配对，必须全部给出。
        self.assertEqual(len(self.candidates), 4)
        self.assertEqual(self.candidates[0].persistence, 5.0)
        self.assertTrue(all(p.persistence >= 0 for p in self.candidates))

    def test_squared_loss_contract_value(self):
        req = TrainingRequest.make(self.identity, target_delta=+1.0, weight=0.5)
        outcome = self.service.training_loss([req], strict_ties=False, ph=self.ph)
        self.assertAlmostEqual(outcome.total_loss, 0.5 * 1.0 ** 2)
        ok = outcome.outcomes[0]
        self.assertEqual(ok.status, "ok")

    def test_gradient_matches_finite_difference_on_fixed_branch(self):
        req = TrainingRequest.make(self.identity, target_delta=+1.0, weight=1.0)
        outcome = self.service.training_loss([req], strict_ties=False, ph=self.ph)
        coords, grads = outcome.as_numpy()
        eps = 1e-6
        for (r, c), analytic in zip(coords.tolist(), grads.tolist()):
            step = eps if analytic > 0 else -eps
            moved = bump(self.model, r, c, step)
            ph2 = run_persistence(moved.build_complex())
            pair2 = ph2.pair_for_identity(self.identity[0], self.identity[1])
            target = self.chosen.persistence + 1.0
            loss_moved = (pair2.persistence - target) ** 2
            fd = (loss_moved - outcome.total_loss) / step
            self.assertAlmostEqual(analytic, fd, places=5)

    def test_overlap_point_is_one_variable_not_two(self):
        # 出生边的关键顶点 (3,5)? 实际出生关键顶点含 (3,4)：它是 L/R 重叠列。
        req = TrainingRequest.make(self.identity, target_delta=-2.0, weight=1.0)
        outcome = self.service.training_loss([req], strict_ties=False, ph=self.ph)
        points = {p for p, _ in outcome.gradient_by_point.items()}
        # 梯度按规范 (行,列) 聚合，重叠格点只出现一次。
        self.assertEqual(len(points), len(set(points)))
        self.assertTrue(any(c == 4 for _, c in points))

    def test_undetermined_skip_on_missing_tile(self):
        partial = ObservationModel.from_record(self.record, delivered=["L"])
        req = TrainingRequest.make(self.identity, target_delta=+1.0)
        outcome = TopoService(partial).training_loss([req], strict_ties=False)
        self.assertEqual(outcome.outcomes[0].status, "skipped")
        self.assertEqual(outcome.outcomes[0].skip_reason, SKIP_UNDETERMINED)
        self.assertEqual(outcome.total_loss, 0.0)

    def test_identity_mismatch_skip(self):
        bad = TrainingRequest.make(((DIM_E, 9, 9, 0), (DIM_F, 9, 9, 0)), target_delta=1.0)
        outcome = self.service.training_loss([bad], ph=self.ph)
        self.assertEqual(outcome.outcomes[0].skip_reason, SKIP_IDENTITY)

    def test_tie_risk_skipped_in_strict_mode_but_subgradient_in_relaxed(self):
        req = TrainingRequest.make(self.identity, target_delta=+1.0)
        strict = self.service.training_loss([req], strict_ties=True, ph=self.ph)
        self.assertEqual(strict.outcomes[0].skip_reason, SKIP_TIE)
        relaxed = self.service.training_loss([req], strict_ties=False, ph=self.ph)
        self.assertEqual(relaxed.outcomes[0].status, "ok")
        self.assertTrue(relaxed.outcomes[0].tie_warning)

    def test_shortening_and_lengthening_have_opposite_targets(self):
        lengthen = TrainingRequest.make(self.identity, target_delta=+1.0)
        shorten = TrainingRequest.make(self.identity, target_delta=-1.0)
        out_l = self.service.training_loss([lengthen], strict_ties=False, ph=self.ph)
        out_s = self.service.training_loss([shorten], strict_ties=False, ph=self.ph)
        self.assertAlmostEqual(out_l.total_loss, out_s.total_loss)
        gl = out_l.gradient_by_point
        gs = out_s.gradient_by_point
        for point in gl:
            self.assertAlmostEqual(gl[point], -gs[point])


if __name__ == "__main__":
    unittest.main()
