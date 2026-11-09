"""Exact arc-based MIP from the challenge PDF (Appendix A), solved with SCIP.
Only practical for small instances (vehicle-indexed binaries blow up fast) - we use it
to check that the scalable hybrid method finds the true optimum on small cases.

Graph: depots s_k, pickups p_o, drop-offs d_o.   x[u,v,k] = 1 if rider k drives u -> v.
"""
import sys
from pyscipopt import Model, quicksum
from vrppd import Instance, eval_seq, MAX_TIME, MAX_ORDERS

EPS = 1e-3   # tiny travel-time surcharge: forbids zero-time sub-cycles that dodge the depot


def solve_exact(inst, time_limit=120, verbose=False):
    K, O = range(inst.n_c), range(inst.n_o)
    m = Model(); m.hideOutput(not verbose); m.setParam("limits/time", time_limit)
    S = [("s", k) for k in K]; P = [("p", o) for o in O]; Dl = [("d", o) for o in O]
    V = S + P + Dl
    def loc(v):
        t, i = v
        return inst.cour_loc[i] if t == "s" else (inst.pick[i] if t == "p" else inst.drop[i])
    def Q(v):
        t, i = v
        return inst.q[i] if t == "p" else (-inst.q[i] if t == "d" else 0)
    Qmax = max(inst.cour_cap); M = MAX_TIME + 50

    # arcs  A1 (depot->pickup), A2 (between pickups/drop-offs), A3 (drop-off->depot), A4 (depot loop)
    arcs = {}
    for k in K:
        s = ("s", k)
        for o in O: arcs[(s, ("p", o), k)] = None
        arcs[(s, s, k)] = None
        for o in O: arcs[(("d", o), s, k)] = None
        for u in P + Dl:
            for v in P + Dl:
                if u != v and not (u[0] == "d" and v[0] == "p" and u[1] == v[1]):
                    arcs[(u, v, k)] = None
    x = {a: m.addVar(vtype="B") for a in arcs}
    t = {v: m.addVar(lb=(inst.tw[v[1]] if v[0] == "p" else 0), ub=(0 if v[0] == "s" else MAX_TIME)) for v in V}
    q = {v: m.addVar(lb=0, ub=Qmax) for v in V}
    out_, in_ = {}, {}
    for (u, v, k) in arcs:
        out_.setdefault((u, k), []).append(x[(u, v, k)]); in_.setdefault((v, k), []).append(x[(u, v, k)])
    in_any = {}
    for (u, v, k), var in x.items(): in_any.setdefault(v, []).append(var)

    for v in V:                                              # (2) every vertex visited once
        m.addCons(quicksum(in_any[v]) == 1)
    for v in V:
        for k in K:                                          # (3) flow conservation per rider
            m.addCons(quicksum(out_.get((v, k), [])) == quicksum(in_.get((v, k), [])))
    for o in O:
        for k in K:                                          # (4) same rider picks up and drops off
            m.addCons(quicksum(in_.get((("p", o), k), [])) == quicksum(in_.get((("d", o), k), [])))
    for k in K:                                              # (5) each rider leaves its depot once
        m.addCons(quicksum(out_[(("s", k), k)]) == 1)
        m.addCons(quicksum(v for (u, w, kk), v in x.items() if kk == k and w[0] == "p") <= MAX_ORDERS)
        m.addCons(q[("s", k)] == Qmax - inst.cour_cap[k])    # (9)
    for (u, v, k), var in x.items():
        if v[0] == "s": continue                             # return arcs carry no time/load
        T = inst.T[loc(u)][loc(v)] + EPS
        m.addCons(t[u] + (T + M) * var <= t[v] + M)          # (6) time propagation
        m.addCons(q[u] + (Q(v) + Qmax) * var <= q[v] + Qmax)  # (8) load propagation
    for o in O:                                              # (7) pickup before drop-off
        m.addCons(t[("d", o)] - t[("p", o)] >= inst.T[inst.pick[o]][inst.drop[o]])
    m.setObjective(quicksum(t[d] for d in Dl), "minimize")   # (1) sum of delivery times
    m.optimize()
    if m.getNSols() == 0: return None, None, None
    routes = {}
    for k in K:
        cur, seq = ("s", k), []
        while True:
            nxt = next((v for (u, v, kk), var in x.items() if u == cur and kk == k and m.getVal(var) > 0.5), None)
            if nxt is None or nxt[0] == "s": break
            if nxt[0] in "pd": seq.append(nxt[1])
            cur = nxt
        if seq: routes[k] = seq
    cost = sum(eval_seq(inst, inst.cour_loc[k], inst.cour_cap[k], s) for k, s in routes.items())
    return routes, cost, m.getGap()


if __name__ == "__main__":
    inst = Instance(sys.argv[1])
    r, c, g = solve_exact(inst, float(sys.argv[2]) if len(sys.argv) > 2 else 120)
    print(f"{inst.name}: exact MIP cost = {c}  gap = {g}")
