"""工位复查规则、跨区路径与台账替代语义的测试。"""
import unittest

from persistence_loss_input import load_record
from topocoat import (
    PENDING,
    RELEASE,
    TICKET,
    ObservationModel,
    ReviewLedger,
    ReviewOrder,
    review_region,
    review_workpiece,
)


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_record()
        cls.orders = {
            rid: ReviewOrder.from_record(rid, spec)
            for rid, spec in cls.record["regions"].items()
        }

    def test_full_workpiece_review_ticket_and_release(self):
        model = ObservationModel.from_record(self.record)
        entries, ph = review_workpiece(model, self.orders)
        verdicts = {(e.verdict, e.pair_identity[0]) for e in entries}
        # 持久 5.0 的缺陷环在两个分区都要发单；允许环在两个分区都放行。
        tickets = [e for e in entries if e.verdict == TICKET]
        releases = [e for e in entries if e.verdict == RELEASE]
        self.assertTrue(tickets)
        self.assertTrue(releases)
        for entry in tickets:
            self.assertEqual(entry.persistence, 5.0)
            self.assertIn("L", entry.tile_ids)
            self.assertIn("R", entry.tile_ids)
            # 可定位：过滤区间、配对身份、指纹齐备。
            self.assertEqual(len(entry.pair_identity), 2)
            self.assertEqual(len(entry.fingerprint), 64)

    def test_cross_region_carries_representative_path_and_regions(self):
        model = ObservationModel.from_record(self.record)
        entries, _ = review_workpiece(model, self.orders)
        ticket = next(e for e in entries if e.verdict == TICKET)
        # 代表路径实际穿过两个分区，且路径点是真实边链顶点。
        self.assertEqual(set(ticket.path_regions), {"left", "right"})
        self.assertTrue(ticket.representative_path)
        for r, c in ticket.representative_path:
            self.assertTrue(0 <= r < 6 and 0 <= c < 7)

    def test_threshold_filters_short_bars(self):
        model = ObservationModel.from_record(self.record)
        strict = ReviewOrder("right", self.orders["right"].box, persistence_threshold=9.0)
        entries, _ = review_region(model, strict)
        self.assertFalse(any(e.verdict == TICKET for e in entries))

    def test_pending_when_tile_missing_and_replaced_on_arrival(self):
        partial = ObservationModel.from_record(self.record, delivered=["L"])
        full = ObservationModel.from_record(self.record)
        ledger = ReviewLedger()

        partial_entries, _ = review_workpiece(partial, self.orders)
        events = ledger.update(partial_entries)
        self.assertTrue(all(e["event"] == "new" for e in events))
        self.assertTrue(ledger.pending())
        self.assertFalse(ledger.tickets())

        full_entries, _ = review_workpiece(full, self.orders)
        events = ledger.update(full_entries)
        replace_events = [e for e in events if e["event"] == "certified_replaces_pending"]
        self.assertTrue(replace_events)
        self.assertFalse(ledger.pending())
        # 已交付的可证明结论数量与全量复查一致。
        self.assertEqual(
            sorted((e.verdict, e.entry_key) for e in ledger.entries),
            sorted((e.verdict, e.entry_key) for e in full_entries),
        )

    def test_certified_conclusion_is_locked_against_silent_flip(self):
        model = ObservationModel.from_record(self.record)
        entries, _ = review_workpiece(model, self.orders)
        ledger = ReviewLedger()
        ledger.update(entries)
        before = {e.entry_key: e.verdict for e in ledger.entries}
        # 用另一指纹的同身份条目尝试静默翻转，应被锁定。
        tampered = [
            type(e)(
                entry_key=e.entry_key, region_ids=e.region_ids,
                verdict=PENDING if e.verdict == TICKET else e.verdict,
                pair_identity=e.pair_identity, dim=e.dim,
                birth_value=e.birth_value, death_value=e.death_value,
                persistence=e.persistence, score_range=e.score_range,
                representative_path=e.representative_path, path_regions=e.path_regions,
                tile_ids=e.tile_ids, certified=e.certified, note=e.note,
                fingerprint="x" * 64,
            )
            for e in entries
        ]
        ledger.update(tampered)
        after = {e.entry_key: e.verdict for e in ledger.entries}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
