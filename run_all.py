"""Run the hybrid solver on every instance folder, write result CSVs, summarise.
usage: python run_all.py <training_data_dir> <out_dir> [seconds_per_instance] [workers]"""
import sys, os, time, csv
from multiprocessing import Pool
from vrppd import Instance, write_solution
from solve_hybrid import solve


def work(args):
    folder, out_dir, tl = args
    inst = Instance(folder)
    t0 = time.time()
    res = solve(inst, tl, verbose=False)
    if res[0] is None:
        return inst.name, inst.n_o, inst.n_c, None, time.time() - t0
    write_solution(inst, res[0], os.path.join(out_dir, inst.name + ".csv"))
    return inst.name, inst.n_o, inst.n_c, res[1], time.time() - t0


if __name__ == "__main__":
    data, out = sys.argv[1], sys.argv[2]
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 60
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    os.makedirs(out, exist_ok=True)
    folders = sorted(os.path.join(data, d) for d in os.listdir(data) if os.path.isdir(os.path.join(data, d)))
    with Pool(nw) as p, open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["instance", "orders", "riders", "cost", "seconds"])
        for r in p.imap_unordered(work, [(d, out, tl) for d in folders]):
            w.writerow(r); f.flush(); print(r, flush=True)
