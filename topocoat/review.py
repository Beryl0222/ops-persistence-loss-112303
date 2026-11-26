"""工位复查：工艺分区查询、放行/缺陷判定、跨区代表路径与结果台账。

两种使用方式：

* :func:`review_workpiece` 在全工件（或完整观测范围）上跑一次持久化，再按
  分区 box 归类条——只有这样才能返回跨区循环及其实际经过的分区；
* :func:`review_region` 按分区 box 裁取独立复核，裁取不改变范围边界语义，
  适合单区复核服务，但不承担跨区路径语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .cells import DIM_E
from .engine import PHResult, PersistencePair, run_persistence
from .observation import ObservationModel
from .openings import chain_is_allowed

#: 复查动作。
TICKET = "ticket"        # 发出缺陷复查单
RELEASE = "release"      # 与允许开孔同调等价，放行
PENDING = "pending"      # 仍依赖缺片，只给可证明的部分

#: 未定的可替代原因。
REPLACED = "certified_replaces_pending"


@dataclass(frozen=True)
class ReviewOrder:
    """一个工艺分区的判读规则。"""

    region_id: str
    box: Tuple[int, int, int, int]
    persistence_threshold: float
    score_range: Optional[Tuple[float, float]] = None   # 过滤区间 [lo, hi]，None 为满范围

    @classmethod
    def from_record(cls, region_id: str, spec: Mapping) -> "ReviewOrder":
        box = tuple(int(x) for x in spec["box"])
        rng = spec.get("score_range")
        return cls(
            region_id=region_id,
            box=box,
            persistence_threshold=float(spec["persistence"]),
            score_range=None if rng is None else (float(rng[0]), float(rng[1])),
        )

    def contains_point(self, r: int, c: int) -> bool:
        r0, c0, r1, c1 = self.box
        return r0 <= r < r1 and c0 <= c < c1


@dataclass(frozen=True)
class ReviewEntry:
    """一条可定位到过滤区间、计分身份与配对单元的复核结果。"""

    entry_key: str
    region_ids: Tuple[str, ...]
    verdict: str
    pair_identity: Tuple[tuple, Optional[tuple]]
    dim: int
    birth_value: float
    death_value: Optional[float]
    persistence: Optional[float]
    score_range: Tuple[float, float]
    representative_path: Tuple[Tuple[int, int], ...]
    path_regions: Tuple[str, ...]
    tile_ids: Tuple[str, ...]
    certified: bool
    note: str
    fingerprint: str


def _bar_in_range(pair: PersistencePair, score_range: Optional[Tuple[float, float]]) -> bool:
    if score_range is None:
        return True
    lo, hi = score_range
    return pair.birth_value <= hi and (pair.infinite or pair.death_value >= lo)


def _pair_tile_ids(ph: PHResult, pair: PersistencePair) -> Tuple[str, ...]:
    cd = ph.complex_data
    ids = set(cd.provenance[pair.birth_idx])
    if not pair.infinite:
        ids.update(cd.provenance[pair.death_idx])
    for edge_idx in pair.representative:
        ids.update(cd.provenance[edge_idx])
    return tuple(sorted(ids))


def _make_entry(region_id: str, order: ReviewOrder, ph: PHResult, pair: PersistencePair,
                verdict: str, note: str, fingerprint: str,
                extra_regions: Tuple[str, ...] = ()) -> ReviewEntry:
    cd = ph.complex_data
    identity = pair.identity(cd)
    path = ph.representative_points(pair)
    score_range = order.score_range or (
        min(cell.value for cell in cd.cells),
        max(cell.value for cell in cd.cells),
    )
    return ReviewEntry(
        entry_key=f"{region_id}|{identity[0]}|{identity[1]}",
        region_ids=(region_id, *extra_regions),
        verdict=verdict,
        pair_identity=identity,
        dim=pair.dim,
        birth_value=pair.birth_value,
        death_value=None if pair.infinite else pair.death_value,
        persistence=None if pair.infinite else pair.persistence,
        score_range=score_range,
        representative_path=path,
        path_regions=_regions_of_path(path, {rid: o for rid, o in ((region_id, order),)}),
        tile_ids=_pair_tile_ids(ph, pair),
        certified=pair.certified,
        note=note,
        fingerprint=fingerprint,
    )


def _regions_of_path(path: Sequence[Tuple[int, int]],
                     orders: Mapping[str, ReviewOrder]) -> Tuple[str, ...]:
    hits = []
    for region_id, order in orders.items():
        if any(order.contains_point(r, c) for r, c in path):
            hits.append(region_id)
    return tuple(hits)


def review_workpiece(
    model: ObservationModel,
    orders: Mapping[str, ReviewOrder],
    budget: Optional[int] = None,
) -> Tuple[List[ReviewEntry], PHResult]:
    """全工件复查：一次持久化，条按代表路径经过的分区归类，支持跨区循环。"""
    cd = model.build_complex()
    ph = run_persistence(cd, budget=budget)
    if not ph.complete:
        raise BudgetExhausted(ph.work_used)
    fingerprint = model.fingerprint()
    full_range = (min(cell.value for cell in cd.cells), max(cell.value for cell in cd.cells))
    entries: List[ReviewEntry] = []

    for region_id, order in orders.items():
        region_fully_observed = _region_fully_observed(model, order.box)
        for pair in ph.pairs:
            if pair.dim != DIM_E or not _bar_in_range(pair, order.score_range):
                continue
            if not pair.infinite and pair.persistence < order.persistence_threshold:
                continue
            path = ph.representative_points(pair)
            if not any(order.contains_point(r, c) for r, c in path):
                continue
            identity = pair.identity(cd)
            path_regions = _regions_of_path(path, orders)
            base = _make_entry(region_id, order, ph, pair, PENDING, "", fingerprint)
            # 重写路径分区与计分身份（全工件视角）。
            base = ReviewEntry(
                entry_key=base.entry_key,
                region_ids=path_regions,
                verdict=PENDING,
                pair_identity=identity,
                dim=pair.dim,
                birth_value=pair.birth_value,
                death_value=None if pair.infinite else pair.death_value,
                persistence=None if pair.infinite else pair.persistence,
                score_range=order.score_range or full_range,
                representative_path=path,
                path_regions=path_regions,
                tile_ids=_pair_tile_ids(ph, pair),
                certified=pair.certified,
                note="",
                fingerprint=fingerprint,
            )
            if not region_fully_observed or not pair.certified:
                reason = pair.uncertified_reason or "分区尚未全观测，缺片可能改变拓扑"
                entries.append(_replace(base, verdict=PENDING, note=reason))
                continue
            if chain_is_allowed(cd, pair.representative):
                note = "代表循环由永久开孔边界张成，与允许开孔同调等价"
                entries.append(_replace(base, verdict=RELEASE, note=note))
            else:
                cross = "；跨区循环，代表路径经过分区 " + "/".join(path_regions) if len(path_regions) > 1 else ""
                entries.append(_replace(
                    base,
                    verdict=TICKET,
                    note="孔洞持久度达到分区阈值且不属于允许开孔" + cross,
                ))
    return entries, ph


def review_region(
    model: ObservationModel,
    order: ReviewOrder,
    budget: Optional[int] = None,
) -> Tuple[List[ReviewEntry], PHResult]:
    """单分区裁取复核（不加 halo，保持复核范围边界语义）。"""
    cd = model.build_complex(box=order.box)
    ph = run_persistence(cd, budget=budget)
    if not ph.complete:
        raise BudgetExhausted(ph.work_used)
    fingerprint = model.fingerprint()
    region_fully_observed = _region_fully_observed(model, order.box)
    entries: List[ReviewEntry] = []
    for pair in ph.pairs:
        if pair.dim != DIM_E or not _bar_in_range(pair, order.score_range):
            continue
        if not pair.infinite and pair.persistence < order.persistence_threshold:
            continue
        entry = _make_entry(order.region_id, order, ph, pair, PENDING, "", fingerprint)
        if not region_fully_observed or not pair.certified:
            reason = pair.uncertified_reason or "分区尚未全观测，缺片可能改变拓扑"
            entries.append(_replace(entry, verdict=PENDING, note=reason))
            continue
        if chain_is_allowed(cd, pair.representative):
            entries.append(_replace(entry, verdict=RELEASE,
                                    note="代表循环由永久开孔边界张成，与允许开孔同调等价"))
        else:
            entries.append(_replace(entry, verdict=TICKET,
                                    note="孔洞持久度达到分区阈值且不属于允许开孔"))
    return entries, ph


def _replace(entry: ReviewEntry, verdict: str, note: str) -> ReviewEntry:
    return ReviewEntry(
        entry_key=entry.entry_key,
        region_ids=entry.region_ids,
        verdict=verdict,
        pair_identity=entry.pair_identity,
        dim=entry.dim,
        birth_value=entry.birth_value,
        death_value=entry.death_value,
        persistence=entry.persistence,
        score_range=entry.score_range,
        representative_path=entry.representative_path,
        path_regions=entry.path_regions,
        tile_ids=entry.tile_ids,
        certified=entry.certified,
        note=note,
        fingerprint=entry.fingerprint,
    )


def _region_fully_observed(model: ObservationModel, box: Sequence[int]) -> bool:
    r0, c0, r1, c1 = box
    return all(model.is_observed(r, c) for r in range(r0, r1) for c in range(c0, c1))


class BudgetExhausted(RuntimeError):
    """给定工作预算内未完成化简；调用方只能使用已证明的部分结果。"""

    def __init__(self, work_used: int):
        super().__init__(f"工作预算耗尽，已完成消元 {work_used} 次")
        self.work_used = work_used


class ReviewLedger:
    """复核台账：已交付结论可定位，缺片更新时只替代旧的未定结果。

    已出具的 certified 结论（ticket/release）在身份不变时锁定；收到新分片后，
    只有旧 ``pending`` 被新可证明结论替代、或出现此前不存在的新配对时才更新。
    """

    def __init__(self):
        self._entries: Dict[str, ReviewEntry] = {}
        self._history: List[Dict[str, str]] = []

    @property
    def entries(self) -> Tuple[ReviewEntry, ...]:
        return tuple(self._entries.values())

    @property
    def history(self) -> Tuple[Dict[str, str], ...]:
        return tuple(self._history)

    def get(self, entry_key: str) -> Optional[ReviewEntry]:
        return self._entries.get(entry_key)

    def update(self, entries: Sequence[ReviewEntry]) -> List[Dict[str, str]]:
        events: List[Dict[str, str]] = []
        for entry in entries:
            old = self._entries.get(entry.entry_key)
            if old is None:
                self._entries[entry.entry_key] = entry
                events.append({"entry_key": entry.entry_key, "event": "new", "verdict": entry.verdict})
                continue
            if old.certified and entry.fingerprint != old.fingerprint and entry.verdict != old.verdict:
                events.append({"entry_key": entry.entry_key, "event": "locked_conflict",
                               "old": old.verdict, "new": entry.verdict})
                continue
            if old.verdict == PENDING and entry.verdict != PENDING:
                self._entries[entry.entry_key] = entry
                events.append({"entry_key": entry.entry_key, "event": REPLACED,
                               "old": PENDING, "new": entry.verdict})
            elif entry.verdict != old.verdict or entry.fingerprint != old.fingerprint:
                self._entries[entry.entry_key] = entry
                events.append({"entry_key": entry.entry_key, "event": "refresh",
                               "old": old.verdict, "new": entry.verdict})
        self._history.extend(events)
        return events

    def tickets(self) -> Tuple[ReviewEntry, ...]:
        return tuple(e for e in self._entries.values() if e.verdict == TICKET)

    def pending(self) -> Tuple[ReviewEntry, ...]:
        return tuple(e for e in self._entries.values() if e.verdict == PENDING)

    def releases(self) -> Tuple[ReviewEntry, ...]:
        return tuple(e for e in self._entries.values() if e.verdict == RELEASE)
