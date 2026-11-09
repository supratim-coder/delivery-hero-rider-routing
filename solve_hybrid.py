"""Hybrid heuristic + MIP solver for the DH VRPPD.

Stage 1  Greedy cheapest-insertion  -> a feasible solution fast
Stage 2  Local search (relocate)    -> better solution, and a pool of good routes
Stage 3  Column pool                -> for each order, bundle it with its nearest
                                       neighbours; solve the best visiting order of
                                       each bundle EXACTLY for the closest rider types
Stage 4  Set-partitioning MIP (SCIP) -> pick the best routes, every order exactly once,
                                       each rider type used at most as often as it has riders
"""
import sys, time, os
from itertools import combinations, permutations
from pyscipopt import Model, quicksum
from vrppd import Instance, eval_seq, best_sequence, write_solution, MAX_ORDERS


# ---------------------------------------------------------------- Stage 1 + 2
def best_insertion(inst, loc, cap, seq, o):
    """Cheapest way to insert order o (pickup+drop-off) into seq. -> (new_cost, new_seq)"""
    if len(seq) >= 2 * MAX_ORDERS:
        return None, None
    best_c, best_s = None, None
    n = len(seq)
    for i in range(n + 1):
        for j in range(i, n + 1):
            s = seq[:i] + [o] + seq[i:j] + [o] + seq[j:]
            c = eval_seq(inst, loc, cap, s)
            if c is not None and (best_c is None or c < best_c):
                best_c, best_s = c, s
    return best_c, best_s


def greedy_and_local_search(inst, deadline, pool):
    """Insert orders by ready time; if that dead-ends, retry hardest-orders-first."""
    for hardest_first in (False, True):
        pool.clear()
        res = _greedy_ls(inst, deadline, pool, hardest_first)
        if res is not None:
            return res
    return None


def _greedy_ls(inst, deadline, pool, hardest_first):
    n_c = inst.n_c
    routes = {k: [] for k in range(n_c)}
    cost = {k: 0 for k in range(n_c)}
    # one empty representative per rider class keeps the search small
    def candidates(o):
        seen = set()
        for k in range(n_c):
            if routes[k]:
                yield k
            else:
                key = (inst.cour_loc[k], inst.cour_cap[k])
                if key not in seen:
                    seen.add(key)
                    yield k

    def n_ok(o):   # how many riders could serve this order on their own
        return sum(len(ks) for loc, cap, ks in inst.classes
                   if eval_seq(inst, loc, cap, [o, o]) is not None)
    if hardest_first:
        order = sorted(range(inst.n_o), key=lambda o: (n_ok(o), inst.tw[o]))
    else:
        order = sorted(range(inst.n_o), key=lambda o: inst.tw[o])
    for o in order:
        best = None
        for k in candidates(o):
            c, s = best_insertion(inst, inst.cour_loc[k], inst.cour_cap[k], routes[k], o)
            if c is not None:
                d = c - cost[k]
                if best is None or d < best[0]:
                    best = (d, k, s, c)
        if best is None:
            return None
        _, k, s, c = best
        routes[k], cost[k] = s, c
        pool.add((k, tuple(s)))

    # relocate local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for o in range(inst.n_o):
            if time.time() > deadline:
                break
            k0 = next(k for k in range(n_c) if o in routes[k])
            rem = [x for x in routes[k0] if x != o]
            rem_cost = eval_seq(inst, inst.cour_loc[k0], inst.cour_cap[k0], rem) if rem else 0
            gain_base = cost[k0] - rem_cost
            best = None
            for k in candidates(o):
                if k == k0:
                    continue
                c, s = best_insertion(inst, inst.cour_loc[k], inst.cour_cap[k], routes[k], o)
                if c is None:
                    continue
                delta = (c - cost[k]) - gain_base
                if delta < -1e-9 and (best is None or delta < best[0]):
                    best = (delta, k, s, c)
            if best:
                _, k, s, c = best
                routes[k0], cost[k0] = rem, rem_cost
                routes[k], cost[k] = s, c
                pool.add((k, tuple(s)))
                if rem:
                    pool.add((k0, tuple(rem)))
                improved = True
    return routes, sum(cost.values())


# ---------------------------------------------------------------- Stage 3
def build_columns(inst, deadline, n_nb=7, n_cls=5):
    """columns[(class_idx, frozenset(bundle))] = (cost, seq)"""
    T = inst.T
    cols = {}
    cls = inst.classes

    def add(bundle, class_ids):
        for ci in class_ids:
            key = (ci, frozenset(bundle))
            if key in cols:
                continue
            loc, cap, _ = cls[ci]
            c, s = best_sequence(inst, loc, cap, bundle)
            if c is not None:
                cols[key] = (c, s)

    def near_classes(bundle):
        sc = sorted(range(len(cls)),
                    key=lambda ci: min(T[cls[ci][0]][inst.pick[o]] for o in bundle))
        return sc[:n_cls]

    # neighbours: orders with close pickups AND close drop-offs
    nbrs = []
    for i in range(inst.n_o):
        d = sorted((T[inst.pick[i]][inst.pick[j]] + T[inst.drop[i]][inst.drop[j]]
                    + abs(inst.tw[i] - inst.tw[j]), j)
                   for j in range(inst.n_o) if j != i)
        nbrs.append([j for _, j in d[:n_nb]])

    for i in range(inst.n_o):                      # singles
        add((i,), near_classes((i,)))
    for size in (2, 3, 4):
        for i in range(inst.n_o):
            if time.time() > deadline:
                return cols
            lim = n_nb if size < 4 else min(n_nb, 5)
            for rest in combinations(nbrs[i][:lim], size - 1):
                b = (i,) + rest
                add(b, near_classes(b))
    return cols


# ---------------------------------------------------------------- Stage 4
def solve_mip(inst, cols, start_keys, time_limit, verbose=False):
    m = Model()
    m.hideOutput(not verbose)
    m.setParam("limits/time", time_limit)
    keys = list(cols)
    y = {}
    for key in keys:
        ub = len(inst.classes[key[0]][2])
        y[key] = m.addVar(vtype="B" if ub == 1 else "I", lb=0, ub=1, obj=cols[key][0],
                          name=f"y_{key[0]}_{len(y)}")
    cover = {o: [] for o in range(inst.n_o)}
    by_class = {}
    for key in keys:
        for o in key[1]:
            cover[o].append(y[key])
        by_class.setdefault(key[0], []).append(y[key])
    for o in range(inst.n_o):
        m.addCons(quicksum(cover[o]) == 1, name=f"cover_{o}")
    for ci, vs in by_class.items():
        m.addCons(quicksum(vs) <= len(inst.classes[ci][2]), name=f"riders_{ci}")
    m.setObjective(quicksum(cols[k][0] * y[k] for k in keys), "minimize")
    sol = m.createSol()
    for k in start_keys:
        m.setSolVal(sol, y[k], 1)
    m.addSol(sol)
    m.optimize()
    if m.getNSols() == 0:
        return None, None, None
    chosen = [k for k in keys if m.getVal(y[k]) > 0.5]
    return chosen, m.getObjVal(), m.getGap()


def assign_to_riders(inst, chosen, cols):
    used = {ci: list(inst.classes[ci][2]) for ci in range(len(inst.classes))}
    routes = {}
    for (ci, b) in chosen:
        routes[used[ci].pop(0)] = cols[(ci, b)][1]
    return routes


def solve(inst, total_time=60.0, verbose=True):
    t0 = time.time()
    pool = set()
    res = greedy_and_local_search(inst, t0 + 0.25 * total_time, pool)
    if res is None:
        return None, None, None
    ls_routes, ls_cost = res
    if verbose:
        print(f"  heuristic (insertion+relocate): {ls_cost}  [{time.time()-t0:.1f}s]")
    cols = build_columns(inst, t0 + 0.5 * total_time)
    # add the heuristic routes (re-sequenced optimally) + remember them as warm start
    cls_idx = {(loc, cap): i for i, (loc, cap, _) in enumerate(inst.classes)}
    start, taken = [], {}
    for k, s in ls_routes.items():
        if not s:
            continue
        ci = cls_idx[(inst.cour_loc[k], inst.cour_cap[k])]
        b = frozenset(s)
        key = (ci, b)
        if key not in cols:
            c, sq = best_sequence(inst, inst.classes[ci][0], inst.classes[ci][1], b)
            cols[key] = (c, sq)
        start.append(key)
    for (k, s) in pool:                             # all routes seen during local search
        ci = cls_idx[(inst.cour_loc[k], inst.cour_cap[k])]
        key = (ci, frozenset(s))
        if key not in cols:
            c, sq = best_sequence(inst, inst.classes[ci][0], inst.classes[ci][1], key[1])
            if c is not None:
                cols[key] = (c, sq)
    if verbose:
        print(f"  column pool: {len(cols)} routes  [{time.time()-t0:.1f}s]")
    left = max(5.0, total_time - (time.time() - t0))
    chosen, obj, gap = solve_mip(inst, cols, start, left)
    if chosen is None or obj > ls_cost + 1e-6:
        return ls_routes, ls_cost, None
    routes = assign_to_riders(inst, chosen, cols)
    if verbose:
        print(f"  SCIP set-partitioning: {obj:.0f}  (gap {gap*100:.2f}%)  [{time.time()-t0:.1f}s]")
    return routes, obj, gap


if __name__ == "__main__":
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 60
    inst = Instance(folder)
    print(f"{inst.name}: {inst.n_c} riders ({len(inst.classes)} types), {inst.n_o} orders")
    routes, obj, gap = solve(inst, tl)
    if routes is None:
        print("  no feasible solution found"); sys.exit(1)
    if out:
        write_solution(inst, routes, out)
