import cvxpy as cp
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm



def solve_for_K(K, V, camera_positions, cells, solver="GUROBI"):
    x = cp.Variable(len(camera_positions), boolean=True)
    y = cp.Variable(len(cells), boolean=True)

    constraints = [cp.sum(x) <= K]
    rooms = sorted(set(room for room, _ in camera_positions))
    for r in rooms:
        idxs = [i for i, (room, _) in enumerate(camera_positions) if room == r]
        constraints.append(cp.sum(x[idxs]) >= 1)

    for j in range(len(cells)):
        cam_idxs = [i for i in range(len(camera_positions)) if V[i, j] == 1]
        if cam_idxs:
            constraints.append(y[j] <= cp.sum([x[i] for i in cam_idxs]))
        else:
            constraints.append(y[j] == 0)

    prob = cp.Problem(cp.Maximize(cp.sum(y)), constraints)

    try:
        if solver == "GUROBI":
            prob.solve(solver=cp.GUROBI)
        elif solver == "CBC":
            prob.solve(solver=cp.CBC)
        elif solver == "GLPK_MI":
            prob.solve(solver=cp.GLPK_MI)
        else:
            raise ValueError("Unknown solver")
    except Exception as e:
        print(f"{solver} solve failed for K={K}: {e}")
        return (K, 0, None)

    coverage = int(np.round(prob.value))
    percent = 100 * coverage / len(cells)
    print(f"K={K}: Coverage = {percent:.2f}%")
    return (K, coverage, x.value)

if __name__ == "__main__":
    print("Loading full visibility matrix...")
    data = np.load("global_selected_visibility_final.npz", allow_pickle=True)
    V_full = data["V"]
    camera_positions_full = data["camera_positions"]
    cells_full = data["cells"]

    print("Reducing problem size...")
    V, camera_positions, cells = V_full, camera_positions_full, cells_full

    Ks = list(range(21, 41))
    max_workers = min(cpu_count(), 4)
    solver = "GUROBI"  # or "CBC"

    print(f"Solving {len(Ks)} values of K using {solver} with {max_workers} workers\n")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(solve_for_K, K, V, camera_positions, cells, solver): K for K in Ks}
        for future in tqdm(as_completed(futures), total=len(Ks), desc="Solving K values"):
            results.append(future.result())

    results.sort(key=lambda x: x[0])

    save_data = {
        "results": results,
        "num_cells": len(cells),
        "npz_file": "global_selected_visibility_final.npz"
    }

    out_name = f"{solver}_reduced_async_Final2.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(save_data, f)

    print(f"\nAll done! Results saved to: {out_name}")
