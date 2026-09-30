#!/usr/bin/env python3
"""Delay-optimizing router for the M3D routing challenge.

The objective is the sum, over every sink, of the driver-to-sink path delay.
A shortest-path tree minimizes that sum for one net; congestion is only a
feasibility constraint (each vertex belongs to at most one net).

The solver:
  1. Builds several legal routings (PathFinder with additive and multiplicative
     congestion, plus the reference negotiated router on a few net orders).
  2. Polishes each net to its exact residual shortest-path tree.
  3. Rips up a net together with the nets blocking a cheaper path, and keeps
     the move only when the group's total delay falls.
  4. Runs large-neighborhood search: re-route small sets of high-delay nets
     from scratch inside the residual grid.

Everything is deterministic given the budget. Scoring is left to m3d.checker.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from m3d.checker import check  # noqa: E402
from m3d.model import Instance, NetRoute, Submission  # noqa: E402
from m3d.negotiated import route_negotiated  # noqa: E402

Rec = Tuple[Tuple[int, ...], Tuple[int, ...], int]  # verts, eids, delay
INF = 1e100


class Router:
    def __init__(self, inst: Instance):
        self.inst = inst
        self.W = inst.width
        self.H = inst.height
        self.L = inst.layers
        self.WH = self.W * self.H
        self.N = self.WH * self.L
        self.adj: List[List[Tuple[int, int, int]]] = [[] for _ in range(self.N)]
        self.eu: List[int] = []
        self.ev: List[int] = []
        self.ed: List[int] = []
        self._build_adj()
        self.ek: Dict[Tuple[int, int], int] = {}
        for eid, (a, b) in enumerate(zip(self.eu, self.ev)):
            self.ek[(a, b) if a < b else (b, a)] = eid

        self.net_pins: Dict[int, List[int]] = {}
        self.pin_owner: Dict[int, int] = {}
        by_id = inst.pin_by_id()
        for n in inst.nets:
            pins = [self.vid(by_id[p].x, by_id[p].y, by_id[p].z) for p in n.pins()]
            self.net_pins[n.id] = pins
            for v in pins:
                self.pin_owner[v] = n.id
        self.forbid: Dict[int, bytearray] = {}
        for n in inst.nets:
            b = bytearray(self.N)
            for v, owner in self.pin_owner.items():
                if owner != n.id:
                    b[v] = 1
            self.forbid[n.id] = b

        self.v_owner = [-1] * self.N
        self.routes: Dict[int, Rec] = {}
        self.stamp = [0] * self.N
        self.closed = [0] * self.N
        self.dist = [0.0] * self.N
        self.parent = [-1] * self.N
        self.pdelay = [0] * self.N
        self.peid = [-1] * self.N
        self.tick = 0
        self.lb: Dict[int, int] = {}
        self.bbox: Dict[int, Tuple[int, int, int, int]] = {}
        for nid, pins in self.net_pins.items():
            xs, ys = [], []
            for v in pins:
                x, y, _z = self.coord(v)
                xs.append(x)
                ys.append(y)
            self.bbox[nid] = (min(xs), min(ys), max(xs), max(ys))
        self._compute_lb()

    def vid(self, x: int, y: int, z: int) -> int:
        return (z * self.H + y) * self.W + x

    def coord(self, v: int) -> Tuple[int, int, int]:
        z, r = divmod(v, self.WH)
        y, x = divmod(r, self.W)
        return (x, y, z)

    def _build_adj(self) -> None:
        W, H, L = self.W, self.H, self.L
        via = self.inst.via_delay
        for z in range(L):
            ld = self.inst.layer_delay[z]
            for y in range(H):
                for x in range(W):
                    u = (z * H + y) * W + x
                    if x + 1 < W:
                        self._link(u, u + 1, ld)
                    if y + 1 < H:
                        self._link(u, u + W, ld)
                    if z + 1 < L:
                        self._link(u, u + self.WH, via)

    def _link(self, u: int, v: int, delay: int) -> None:
        eid = len(self.eu)
        self.eu.append(u)
        self.ev.append(v)
        self.ed.append(delay)
        self.adj[u].append((v, delay, eid))
        self.adj[v].append((u, delay, eid))

    def _compute_lb(self) -> None:
        """Unconstrained shortest-path-tree delay (other pins forbidden)."""
        saved_owner = self.v_owner
        self.v_owner = [-1] * self.N
        for nid in self.net_pins:
            rec = self.route(nid, mode=0)
            if rec is None:
                raise RuntimeError(f"net {nid} is disconnected even without congestion")
            self.lb[nid] = rec[2]
        self.v_owner = saved_owner

    # -- occupancy -----------------------------------------------------------
    def clear(self) -> None:
        self.v_owner = [-1] * self.N
        self.routes.clear()

    def commit(self, nid: int, rec: Rec) -> None:
        for v in rec[0]:
            if self.v_owner[v] != -1:
                raise RuntimeError(f"commit conflict net {nid} at {v} owner {self.v_owner[v]}")
            self.v_owner[v] = nid
        self.routes[nid] = rec

    def remove(self, nid: int) -> None:
        rec = self.routes.pop(nid)
        for v in rec[0]:
            if self.v_owner[v] != nid:
                raise RuntimeError(f"remove mismatch net {nid} at {v}")
            self.v_owner[v] = -1

    def total(self) -> int:
        return sum(r[2] for r in self.routes.values())

    def snapshot(self) -> Dict[int, Rec]:
        return dict(self.routes)

    def restore(self, saved: Dict[int, Rec]) -> None:
        """Replace the entire routing with a full snapshot."""
        self.clear()
        for nid, rec in saved.items():
            self.commit(nid, rec)

    # -- core search ---------------------------------------------------------
    def route(self, nid: int, mode: int, pres: float = 0.0, penalty: int = 0,
              h: Optional[List[float]] = None, soft: Optional[List[int]] = None
              ) -> Optional[Rec]:
        """Shortest-path tree from the driver.

        mode 0: hard — vertices owned by another net are forbidden; cost = delay.
        mode 1: penalty — owned vertices are usable at ``penalty`` extra cost.
        mode 2: soft additive — ``soft`` counts (group) congestion;
                vertices with v_owner != -1 are forbidden (outside the group).
        mode 3: soft multiplicative — cost = delay * (1 + h + pres * soft).
        """
        self.tick += 1
        if self.tick >= 2_000_000_000:
            self.stamp = [0] * self.N
            self.closed = [0] * self.N
            self.tick = 1
        t = self.tick
        pins = self.net_pins[nid]
        driver = pins[0]
        sinks = pins[1:]
        sinkset = set(sinks)
        need = len(sinkset)
        stamp = self.stamp
        closed = self.closed
        dist = self.dist
        parent = self.parent
        pdelay = self.pdelay
        peid = self.peid
        forbid = self.forbid[nid]
        owner = self.v_owner
        adj = self.adj
        stamp[driver] = t
        dist[driver] = 0.0
        parent[driver] = -1
        heap: List[Tuple[float, int]] = [(0.0, driver)]
        got = 0
        heappush = heapq.heappush
        heappop = heapq.heappop
        while heap and got < need:
            d, u = heappop(heap)
            if closed[u] == t:
                continue
            if stamp[u] != t or d > dist[u]:
                continue
            closed[u] = t
            if u in sinkset and u != driver:
                got += 1
                if got == need:
                    break
            du = dist[u]
            for v, w, eid in adj[u]:
                if forbid[v]:
                    continue
                if mode == 0:
                    if owner[v] != -1:
                        continue
                    nd = du + w
                elif mode == 1:
                    extra = penalty if owner[v] != -1 else 0
                    nd = du + w + extra
                elif mode == 2:
                    if owner[v] != -1:
                        continue
                    nd = du + w + pres * soft[v] + h[v]
                else:  # mode 3
                    if owner[v] != -1:
                        continue
                    nd = du + w * (1.0 + h[v] + pres * soft[v])
                if stamp[v] != t or nd < dist[v]:
                    stamp[v] = t
                    dist[v] = nd
                    parent[v] = u
                    pdelay[v] = w
                    peid[v] = eid
                    heappush(heap, (nd, v))
        if got < need:
            return None
        return self._materialize(driver, sinks, t)

    def _materialize(self, driver: int, sinks: Sequence[int], t: int) -> Optional[Rec]:
        parent = self.parent
        pdelay = self.pdelay
        peid = self.peid
        stamp = self.stamp
        eids: List[int] = []
        seen_e = set()
        verts = [driver]
        seen_v = {driver}
        node_d = {driver: 0}
        for s in sinks:
            chain: List[int] = []
            cur = s
            guard = 0
            while cur not in node_d:
                if stamp[cur] != t or parent[cur] < 0:
                    return None
                chain.append(cur)
                cur = parent[cur]
                guard += 1
                if guard > self.N:
                    return None
            acc = node_d[cur]
            for node in reversed(chain):
                acc += pdelay[node]
                node_d[node] = acc
                if node not in seen_v:
                    seen_v.add(node)
                    verts.append(node)
            cur = s
            while cur != driver:
                eid = peid[cur]
                if eid not in seen_e:
                    seen_e.add(eid)
                    eids.append(eid)
                cur = parent[cur]
        if len(eids) != len(verts) - 1:
            return None
        delay = 0
        for s in sinks:
            delay += node_d[s]
        return (tuple(verts), tuple(eids), delay)

    # -- group routers (nets must currently be removed) ----------------------
    def reroute_hard(self, nids: Sequence[int]) -> bool:
        placed: List[int] = []
        for nid in nids:
            rec = self.route(nid, mode=0)
            if rec is None:
                for p in placed:
                    self.remove(p)
                return False
            self.commit(nid, rec)
            placed.append(nid)
        return True

    def reroute_soft(self, nids: Sequence[int], mode: int = 2, max_iters: int = 25,
                     pres0: float = 0.5, pres_mult: float = 1.7,
                     hist: float = 0.5, rescue: bool = False) -> bool:
        """PathFinder inside ``nids``. Outside nets stay committed and are hard blocks.

        On success the group is committed. On failure nothing from the group is
        committed. With ``rescue``, a small leftover conflict is cleared by
        ripping up the nets around it and negotiating that subset again.
        """
        if not nids:
            return True
        soft = [0] * self.N
        h = [0.0] * self.N
        placed: Dict[int, Rec] = {}
        pres = pres0
        for nid in nids:
            rec = self.route(nid, mode=mode, pres=pres, h=h, soft=soft)
            if rec is None:
                return False
            for v in rec[0]:
                soft[v] += 1
            placed[nid] = rec
        for _it in range(max_iters):
            affected = []
            for nid in nids:
                rec = placed[nid]
                if any(soft[v] > 1 for v in rec[0]):
                    affected.append(nid)
            if not affected:
                for nid in nids:
                    self.commit(nid, placed[nid])
                return True
            seen = set()
            for nid in affected:
                for v in placed[nid][0]:
                    c = soft[v]
                    if c > 1 and v not in seen:
                        seen.add(v)
                        h[v] += hist * (c - 1)
            pres = pres * pres_mult
            if pres > 1e6:
                pres = 1e6
            for nid in nids:
                if nid not in affected and nid in placed:
                    # affected list is the only ones to reroute; membership test
                    pass
            affected_set = set(affected)
            for nid in nids:
                if nid not in affected_set:
                    continue
                old = placed[nid]
                for v in old[0]:
                    soft[v] -= 1
                rec = self.route(nid, mode=mode, pres=pres, h=h, soft=soft)
                if rec is None:
                    for v in old[0]:
                        soft[v] += 1
                    continue
                for v in rec[0]:
                    soft[v] += 1
                placed[nid] = rec
        if any(soft[v] > 1 for v in range(self.N)):
            if rescue and self._rescue(nids, placed, soft, mode):
                return True
            return False
        for nid in nids:
            self.commit(nid, placed[nid])
        return True

    def _rescue(self, nids: Sequence[int], placed: Dict[int, Rec], soft: List[int],
                mode: int) -> bool:
        """Rip up nets near the remaining conflicts and negotiate just those."""
        conflicts = [v for v in range(self.N) if soft[v] > 1]
        if not conflicts or len(conflicts) > 24:
            return False
        for dist in (1, 2, 3, 5):
            zone = set()
            for v in conflicts:
                x, y, _z = self.coord(v)
                y0 = max(0, y - dist)
                y1 = min(self.H - 1, y + dist)
                x0 = max(0, x - dist)
                x1 = min(self.W - 1, x + dist)
                for zz in range(self.L):
                    for yy in range(y0, y1 + 1):
                        base = (zz * self.H + yy) * self.W
                        for xx in range(x0, x1 + 1):
                            zone.add(base + xx)
            rip = [nid for nid in nids if any(v in zone for v in placed[nid][0])]
            if len(rip) < 2 or len(rip) > max(40, len(nids) // 2):
                continue
            ripset = set(rip)
            keep = [nid for nid in nids if nid not in ripset]
            committed: List[int] = []
            try:
                for nid in keep:
                    self.commit(nid, placed[nid])
                    committed.append(nid)
            except RuntimeError:
                for nid in committed:
                    self.remove(nid)
                continue
            ok = self.reroute_soft(
                rip, mode=mode, max_iters=30, pres0=0.7, pres_mult=1.85,
                hist=0.55, rescue=False,
            )
            if ok:
                return True
            for nid in committed:
                self.remove(nid)
        return False

    def pathfinder_all(self, order: Sequence[int], mode: int = 2, max_iters: int = 40,
                       pres0: float = 0.5, pres_mult: float = 1.7,
                       hist: float = 0.5) -> bool:
        self.clear()
        return self.reroute_soft(order, mode=mode, max_iters=max_iters,
                                 pres0=pres0, pres_mult=pres_mult, hist=hist,
                                 rescue=True)

    # -- improvement ---------------------------------------------------------
    def polish(self, passes: int = 10) -> int:
        """Re-route each net on the residual grid. Returns total delay removed."""
        nids = list(self.routes)
        removed = 0
        for _ in range(passes):
            gain = 0
            for nid in sorted(nids, key=lambda n: -self.routes[n][2]):
                old = self.routes[nid]
                self.remove(nid)
                neu = self.route(nid, mode=0)
                if neu is not None and (
                    neu[2] < old[2] or (neu[2] == old[2] and len(neu[0]) < len(old[0]))
                ):
                    self.commit(nid, neu)
                    gain += old[2] - neu[2]
                else:
                    self.commit(nid, old)
            removed += gain
            if gain == 0:
                break
        return removed

    def _blockers(self, rec: Rec, nid: int) -> Tuple[int, ...]:
        found = set()
        owner = self.v_owner
        for v in rec[0]:
            o = owner[v]
            if o != -1 and o != nid:
                found.add(o)
        return tuple(sorted(found))

    def try_group(self, nid: int, rec: Rec, blockers: Sequence[int],
                  allow_soft: bool) -> int:
        """Fix ``nid`` on ``rec`` and re-route ``blockers``. Keep only a strict improvement."""
        blockers = [b for b in blockers if b != nid]
        if not blockers:
            return 0
        group = [nid] + blockers
        saved = {g: self.routes[g] for g in group}
        old_sum = sum(saved[g][2] for g in group)
        for g in group:
            self.remove(g)
        if any(self.v_owner[v] != -1 for v in rec[0]):
            for g in group:
                self.commit(g, saved[g])
            return 0
        self.commit(nid, rec)
        order = sorted(blockers, key=lambda b: -saved[b][2])
        ok = self.reroute_hard(order)
        if not ok and allow_soft:
            ok = self.reroute_soft(order, mode=2, max_iters=18, pres0=0.8, pres_mult=1.8)
        if not ok:
            if nid in self.routes and self.v_owner[rec[0][0]] == nid:
                self.remove(nid)
            for g in group:
                self.commit(g, saved[g])
            return 0
        new_sum = sum(self.routes[g][2] for g in group)
        if new_sum < old_sum:
            return old_sum - new_sum
        for g in group:
            self.remove(g)
        for g in group:
            self.commit(g, saved[g])
        return 0

    def group_search(self, rounds: int = 3, max_b: int = 8, soft_cap: int = 12) -> int:
        penalties = (0, 1, 2, 4, 8, 16, 32, 64)
        total_gain = 0
        soft_used = 0
        for _rnd in range(rounds):
            gain = 0
            order = sorted(self.routes, key=lambda n: -(self.routes[n][2] - self.lb[n]))
            for nid in order:
                old = self.routes[nid]
                slack = old[2] - self.lb[nid]
                if slack <= 0:
                    continue
                self.remove(nid)
                cands = []
                seen = set()
                best_free: Optional[Rec] = None
                for pen in penalties:
                    rec = self.route(nid, mode=1, penalty=pen)
                    if rec is None:
                        continue
                    bl = self._blockers(rec, nid)
                    if not bl:
                        if best_free is None or rec[2] < best_free[2] or (
                            rec[2] == best_free[2] and len(rec[0]) < len(best_free[0])
                        ):
                            best_free = rec
                        continue
                    if bl in seen or len(bl) > max_b:
                        continue
                    if rec[2] > old[2] + max(40, old[2] // 4):
                        continue
                    seen.add(bl)
                    cands.append((rec, bl))
                chosen = old
                if best_free is not None and (
                    best_free[2] < old[2] or (
                        best_free[2] == old[2] and len(best_free[0]) < len(old[0])
                    )
                ):
                    chosen = best_free
                self.commit(nid, chosen)
                gain += old[2] - chosen[2]
                # Prefer candidates that cut this net's delay the most.
                cands.sort(key=lambda c: (c[0][2], len(c[1])))
                for rec, bl in cands[:3]:
                    allow_soft = soft_used < soft_cap and len(bl) <= 5 and slack >= 20
                    if allow_soft:
                        soft_used += 1
                    g = self.try_group(nid, rec, bl, allow_soft=allow_soft)
                    gain += g
            total_gain += gain
            if gain == 0:
                break
        return total_gain

    def pair_pass(self, max_pairs: int = 250, rng: Optional[random.Random] = None) -> int:
        rng = rng or random.Random(0)
        nids = [n for n in self.routes if self.routes[n][2] > self.lb[n]]
        nids.sort(key=lambda n: -(self.routes[n][2] - self.lb[n]))
        pairs = []
        boxes = self.bbox
        for i, a in enumerate(nids):
            ax0, ay0, ax1, ay1 = boxes[a]
            for b in nids[i + 1:]:
                bx0, by0, bx1, by1 = boxes[b]
                if ax1 + 2 < bx0 or bx1 + 2 < ax0 or ay1 + 2 < by0 or by1 + 2 < ay0:
                    continue
                pairs.append((a, b))
        rng.shuffle(pairs)
        gain = 0
        for a, b in pairs[:max_pairs]:
            saved = {a: self.routes[a], b: self.routes[b]}
            old = saved[a][2] + saved[b][2]
            best = old
            best_state = None
            for order in ((a, b), (b, a)):
                self.remove(a)
                self.remove(b)
                ok = self.reroute_hard(order)
                if not ok:
                    self.commit(a, saved[a])
                    self.commit(b, saved[b])
                    continue
                new = self.routes[a][2] + self.routes[b][2]
                if new < best:
                    best = new
                    best_state = {a: self.routes[a], b: self.routes[b]}
                self.remove(a)
                self.remove(b)
                self.commit(a, saved[a])
                self.commit(b, saved[b])
            if best_state is not None:
                self.remove(a)
                self.remove(b)
                self.commit(a, best_state[a])
                self.commit(b, best_state[b])
                gain += old - best
        return gain

    def lns(self, steps: int, rng: random.Random, soft_every: int = 4) -> int:
        nids = list(self.routes)
        if not nids:
            return 0
        gain = 0
        for step in range(steps):
            weights = [max(1, self.routes[n][2] - self.lb[n]) for n in nids]
            k = rng.choice((3, 4, 5, 6, 8))
            k = min(k, len(nids))
            pool = list(nids)
            wpool = list(weights)
            group = []
            for _ in range(k):
                total_w = sum(wpool)
                r = rng.randrange(total_w)
                acc = 0
                idx = 0
                for i, w in enumerate(wpool):
                    acc += w
                    if acc > r:
                        idx = i
                        break
                group.append(pool.pop(idx))
                wpool.pop(idx)
            saved = {g: self.routes[g] for g in group}
            old = sum(saved[g][2] for g in group)
            for g in group:
                self.remove(g)
            order = sorted(group, key=lambda g: -saved[g][2])
            ok = self.reroute_hard(order)
            if not ok and step % soft_every == 0:
                ok = self.reroute_soft(order, mode=2, max_iters=16, pres0=0.6, pres_mult=1.8)
            if not ok:
                for g in group:
                    # failure leaves the group uncommitted
                    self.commit(g, saved[g])
                continue
            new = sum(self.routes[g][2] for g in group)
            if new < old:
                gain += old - new
            else:
                for g in group:
                    self.remove(g)
                for g in group:
                    self.commit(g, saved[g])
        return gain

    # -- import / export -----------------------------------------------------
    def load_submission(self, sub: Submission) -> None:
        self.clear()
        for r in sub.routes:
            eids = []
            seen = set()
            verts = set()
            for a, b in r.edges:
                ua = self.vid(a[0], a[1], a[2])
                ub = self.vid(b[0], b[1], b[2])
                eid = self.ek[(ua, ub) if ua < ub else (ub, ua)]
                if eid not in seen:
                    seen.add(eid)
                    eids.append(eid)
                verts.add(ua)
                verts.add(ub)
            for p in self.net_pins[r.net]:
                verts.add(p)
            # delay via a walk over edges
            delay = self._delay_from_edges(r.net, eids)
            self.commit(r.net, (tuple(verts), tuple(eids), delay))

    def _delay_from_edges(self, nid: int, eids: Sequence[int]) -> int:
        adj: Dict[int, List[Tuple[int, int]]] = {}
        pins = self.net_pins[nid]
        driver = pins[0]
        for eid in eids:
            a, b, w = self.eu[eid], self.ev[eid], self.ed[eid]
            adj.setdefault(a, []).append((b, w))
            adj.setdefault(b, []).append((a, w))
        dist = {driver: 0}
        stack = [driver]
        while stack:
            u = stack.pop()
            for v, w in adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + w
                    stack.append(v)
        return sum(dist[s] for s in pins[1:])

    def to_submission(self) -> Submission:
        routes = []
        for n in self.inst.nets:
            _verts, eids, _d = self.routes[n.id]
            edges = []
            for eid in eids:
                edges.append((self.coord(self.eu[eid]), self.coord(self.ev[eid])))
            routes.append(NetRoute(net=n.id, edges=edges))
        return Submission(instance=self.inst.name, routes=routes)


def _orders(inst: Instance, router: Router) -> List[List[int]]:
    nids = [n.id for n in inst.nets]
    def bbox_key(i: int) -> int:
        x0, y0, x1, y1 = router.bbox[i]
        return (x1 - x0) + (y1 - y0)
    by_bbox = sorted(nids, key=lambda i: (-bbox_key(i), i))
    by_id = sorted(nids)
    by_lb = sorted(nids, key=lambda i: (-router.lb[i], i))
    by_bbox_asc = sorted(nids, key=lambda i: (bbox_key(i), i))
    return [by_bbox, by_id, by_lb, by_bbox_asc]


def improve(router: Router, rng: random.Random, budget_s: float, label: str) -> int:
    """Polish + group moves + pairs + LNS until the budget runs out."""
    t0 = time.time()
    router.polish()
    print(f"    {label} polish {router.total()} ({time.time()-t0:.1f}s)", flush=True)
    for round_i in range(6):
        if time.time() - t0 > budget_s:
            break
        g = router.group_search(rounds=2, max_b=8, soft_cap=10)
        router.polish()
        print(f"    {label} group{round_i} +{g} -> {router.total()} ({time.time()-t0:.1f}s)",
              flush=True)
        if time.time() - t0 > budget_s:
            break
        p = router.pair_pass(max_pairs=200, rng=rng)
        router.polish()
        print(f"    {label} pairs +{p} -> {router.total()} ({time.time()-t0:.1f}s)",
              flush=True)
        if time.time() - t0 > budget_s:
            break
        ln = router.lns(steps=28, rng=rng, soft_every=3)
        router.polish()
        print(f"    {label} lns +{ln} -> {router.total()} ({time.time()-t0:.1f}s)",
              flush=True)
        if g == 0 and p == 0 and ln == 0:
            break
    return router.total()


def solve_instance(inst: Instance, budget_s: float = 90.0, seed: int = 1) -> Tuple[Submission, dict]:
    t0 = time.time()
    router = Router(inst)
    orders = _orders(inst, router)
    schedules = (
        (0.5, 1.8, 0.5),
        (0.4, 1.6, 0.4),
        (0.5, 2.2, 0.8),
        (0.5, 1.7, 0.5),
    )
    # (order index, schedule index). Stop once two distinct legal basins exist.
    attempts = ((0, 0), (2, 0), (1, 1), (0, 2), (3, 0), (1, 3))
    polished: List[Tuple[int, str, Dict[int, Rec]]] = []
    for oi, si in attempts:
        if len(polished) >= 2 and time.time() - t0 > budget_s * 0.4:
            break
        if len(polished) >= 3:
            break
        pres0, mult, hist = schedules[si]
        tag = f"pf-o{oi}-s{si}"
        ok = router.pathfinder_all(
            orders[oi], mode=2, max_iters=42, pres0=pres0, pres_mult=mult, hist=hist,
        )
        if not ok:
            print(f"  start {tag} failed ({time.time()-t0:.1f}s)", flush=True)
            continue
        raw = router.total()
        router.polish()
        polished.append((router.total(), tag, router.snapshot()))
        print(f"  start {tag} raw {raw} polish {router.total()} ({time.time()-t0:.1f}s)",
              flush=True)

    # Negotiated-congestion basins. Different net orders land in different
    # local optima; keep them when the clock allows.
    if time.time() - t0 < budget_s * 0.55:
        for order_name in ("bbox_desc", "id"):
            if time.time() - t0 > budget_s * 0.55 and polished:
                break
            sub, stats = route_negotiated(inst, order=order_name, max_iters=40)
            if sub is None:
                print(f"  negotiated {order_name} failed {stats}", flush=True)
                continue
            router.load_submission(sub)
            raw = router.total()
            router.polish()
            polished.append((router.total(), "neg-" + order_name, router.snapshot()))
            print(f"  start neg-{order_name} raw {raw} polish {router.total()} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    if not polished:
        print("  PathFinder did not legalize; falling back to negotiated", flush=True)
        for order_name in ("bbox_desc", "id", "bbox_asc"):
            sub, stats = route_negotiated(inst, order=order_name, max_iters=50)
            if sub is None:
                print(f"  negotiated {order_name} failed {stats}", flush=True)
                continue
            router.load_submission(sub)
            router.polish()
            polished.append((router.total(), "neg-" + order_name, router.snapshot()))
            print(f"  start neg-{order_name} polish {router.total()} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            if len(polished) >= 2:
                break

    if not polished:
        raise RuntimeError(f"no legal start for {inst.name}")

    polished.sort(key=lambda x: x[0])

    remaining = max(10.0, budget_s - (time.time() - t0))
    # Spend most of the budget on the best basin, a slice on the runner-up
    # if it is a different start and close.
    best_delay = None
    best_snap = None
    best_name = None
    n_deep = 1
    if len(polished) > 1 and polished[1][0] <= int(polished[0][0] * 1.03):
        n_deep = 2
    for i in range(n_deep):
        delay0, name, snap = polished[i]
        share = remaining * (0.7 if i == 0 else 0.3) if n_deep == 2 else remaining
        router.restore(snap)
        improve(router, random.Random(seed + 17 + i), share, name)
        d = router.total()
        if best_delay is None or d < best_delay:
            best_delay = d
            best_snap = router.snapshot()
            best_name = name
    assert best_snap is not None
    router.restore(best_snap)
    # Final polish in case the last move opened a single-net improvement.
    router.polish()
    sub = router.to_submission()
    res = check(inst, sub)
    if not res.legal or res.total_delay != router.total():
        raise RuntimeError(
            f"{inst.name} illegal or delay mismatch checker={res.total_delay} "
            f"internal={router.total()} reasons={res.reasons[:6]}"
        )
    meta = {
        "delay": res.total_delay,
        "lb": sum(router.lb.values()),
        "start": best_name,
        "seconds": round(time.time() - t0, 3),
    }
    print(f"  DONE {inst.name} delay={res.total_delay} lb={meta['lb']} "
          f"via {best_name} in {meta['seconds']}s", flush=True)
    return sub, meta


def _suite_cases(suite: str, only: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    man = json.load(open(os.path.join(suite, "suite.json")))
    out = []
    for c in man["cases"]:
        if only and c["name"] not in only:
            continue
        out.append((c["name"], os.path.join(suite, c["instance_file"])))
    return out


def _solve_one(payload: Tuple[str, str, float, int, str]) -> Tuple[str, dict]:
    name, path, budget, seed, out = payload
    print(f"=== {name} ===", flush=True)
    inst = Instance.load(path)
    sub, meta = solve_instance(inst, budget_s=budget, seed=seed)
    sub.save(os.path.join(out, f"{name}.sol.json"))
    return name, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Delay-optimizing M3D router")
    ap.add_argument("--suite", default=os.path.join(ROOT, "benchmarks_hard"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--budget", type=float, default=120.0,
                    help="seconds of search per case")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cases = _suite_cases(args.suite, args.cases)
    payloads = [
        (name, path, args.budget, args.seed + i * 17, args.out)
        for i, (name, path) in enumerate(cases)
    ]
    if args.jobs == 1:
        results = [_solve_one(p) for p in payloads]
    else:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(_solve_one, payloads))
    runtimes = {name: meta["seconds"] for name, meta in results}
    with open(os.path.join(args.out, "runtime.json"), "w") as fh:
        json.dump(runtimes, fh, indent=1)
        fh.write("\n")
    print("wrote", args.out, flush=True)
    for name, meta in results:
        print(f"  {name}: delay {meta['delay']} lb {meta['lb']} via {meta['start']} "
              f"in {meta['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
