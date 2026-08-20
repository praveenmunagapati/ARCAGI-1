import json, time, argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from arc_symbolic_solver import PureSymbolicSolverV17, G, safe, exact

def eval_single(fp_str):
    fp = Path(fp_str)
    task = json.loads(fp.read_text(encoding="utf-8"))
    solver = PureSymbolicSolverV17()
    ts = time.perf_counter()
    sols = solver.solve(task)
    dt = time.perf_counter() - ts
    ti = [G(ex["input"]) for ex in task.get("test", [])]
    to = [G(ex["output"]) for ex in task.get("test", []) if "output" in ex]
    s1 = s2 = False
    if sols:
        if to:
            for si, sol in enumerate(sols[:3]):
                try:
                    p = safe(sol, ti[0])
                    if exact(p, to[0]):
                        if si == 0: s1 = True
                        s2 = True
                        break
                except: pass
    return fp.stem, len(sols) > 0, s1, s2, dt, len(sols)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="evaluation", choices=["evaluation", "training"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = sorted(str(p) for p in Path(f"arc_data/{args.split}").glob("*.json"))
    print(f"Running {args.split.upper()} benchmark on {len(tasks)} tasks with {args.workers} workers...")
    
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(eval_single, tasks))
    total_t = time.perf_counter() - t0

    fit = sum(1 for r in results if r[1])
    s1 = sum(1 for r in results if r[2])
    s2 = sum(1 for r in results if r[3])
    solved_names = [r[0] for r in results if r[2] or r[3]]

    print("\n" + "="*80)
    print(f"FINAL RESULTS: {args.split.upper()} (STRICT NON-LLM)")
    print("="*80)
    print(f"Tasks:       {len(tasks)}")
    print(f"Train Fit:   {fit}/{len(tasks)} ({100*fit/len(tasks):.2f}%)")
    print(f"Solved Top1: {s1}/{len(tasks)} ({100*s1/len(tasks):.2f}%)")
    print(f"Solved Top2: {s2}/{len(tasks)} ({100*s2/len(tasks):.2f}%)")
    print(f"Total Time:  {total_t:.2f}s ({total_t/len(tasks)*1000:.1f}ms/task)")
    print("="*80)
    print(f"\nSolved ({len(solved_names)}): {', '.join(solved_names[:50])}")
    if len(solved_names) > 50:
        print(f"  ... and {len(solved_names)-50} more")
