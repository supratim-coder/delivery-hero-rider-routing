"""Shared data model + route evaluation for the Delivery Hero VRPPD (CO@Work 2024).

Rules (taken from the official feasibility_checker.py):
  * a route is a sequence of order IDs; 1st appearance = pickup, 2nd = drop-off
  * rider waits at a pickup until the order's Time Window Start
  * route ends at the last drop-off; that time must be <= 180 min
  * at most 4 orders (8 stops) per rider, load never above rider capacity
  * cost = sum of drop-off times (rider clock starts at 0 at its start location)
"""
import csv
import os
from itertools import combinations

MAX_TIME = 180
MAX_ORDERS = 4


class Instance:
    def __init__(self, folder):
        self.name = os.path.basename(os.path.normpath(folder))
        with open(os.path.join(folder, "couriers.csv")) as f:
            rows = list(csv.DictReader(f))
        self.cour_id = [int(r["ID"]) for r in rows]
        self.cour_loc = [int(r["Location"]) - 1 for r in rows]
        self.cour_cap = [int(r["Capacity"]) for r in rows]
        with open(os.path.join(folder, "deliveries.csv")) as f:
            rows = list(csv.DictReader(f))
        self.ord_id = [int(r["ID"]) for r in rows]
        self.q = [int(r["Capacity"]) for r in rows]
        self.pick = [int(r["Pickup Loc"]) - 1 for r in rows]
        self.tw = [int(r["Time Window Start"]) for r in rows]
        self.stack = [int(r["Pickup Stacking_Id"]) for r in rows]
        self.drop = [int(r["Dropoff Loc"]) - 1 for r in rows]
        with open(os.path.join(folder, "traveltimes.csv")) as f:
            rd = csv.reader(f)
            next(rd)
            self.T = [[int(v) for v in r[1:]] for r in rd]
        self.n_c, self.n_o = len(self.cour_id), len(self.ord_id)
        # rider classes: riders with the same (start, capacity) are interchangeable
        cls = {}
        self.cls_of = []
        for k in range(self.n_c):
            key = (self.cour_loc[k], self.cour_cap[k])
            cls.setdefault(key, []).append(k)
        self.classes = [(loc, cap, ks) for (loc, cap), ks in cls.items()]


def eval_seq(inst, loc, cap, seq):
    """Cost of a rider route (sum of drop-off times) or None if infeasible."""
    T, q, pick, drop, tw = inst.T, inst.q, inst.pick, inst.drop, inst.tw
    t = cost = load = 0
    last = loc
    inbag = set()
    for o in seq:
        if o in inbag:
            inbag.discard(o)
            t += T[last][drop[o]]
            last = drop[o]
            cost += t
            load -= q[o]
        else:
            inbag.add(o)
            load += q[o]
            if load > cap:
                return None
            a = t + T[last][pick[o]]
            t = a if a > tw[o] else tw[o]
            last = pick[o]
    return cost if t <= MAX_TIME else None


def best_sequence(inst, loc, cap, bundle):
    """Exact best visiting order for a small set of orders (<=4) by branch & bound.
    Returns (cost, seq) or (None, None)."""
    T, q, pick, drop, tw = inst.T, inst.q, inst.pick, inst.drop, inst.tw
    bundle = tuple(bundle)
    n = len(bundle)
    best = [10 ** 9, None]
    seq = []

    def rec(t, last, load, picked, done, cost):
        if cost >= best[0]:
            return
        if done == (1 << n) - 1:
            best[0], best[1] = cost, list(seq)
            return
        for i in range(n):
            bit = 1 << i
            o = bundle[i]
            if not picked & bit:
                nl = load + q[o]
                if nl > cap:
                    continue
                a = t + T[last][pick[o]]
                nt = a if a > tw[o] else tw[o]
                if nt > MAX_TIME:
                    continue
                seq.append(o)
                rec(nt, pick[o], nl, picked | bit, done, cost)
                seq.pop()
            elif not done & bit:
                nt = t + T[last][drop[o]]
                if nt > MAX_TIME:
                    continue
                seq.append(o)
                rec(nt, drop[o], load - q[o], picked, done | bit, cost + nt)
                seq.pop()

    rec(0, loc, 0, 0, 0, 0)
    return (best[0], best[1]) if best[1] is not None else (None, None)


def write_solution(inst, routes, path):
    """routes: dict courier_index -> sequence of order indices."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Route"])
        for k in range(inst.n_c):
            w.writerow([inst.cour_id[k]] + [inst.ord_id[o] for o in routes.get(k, [])])
