import torch
import argparse
from copy import deepcopy
import time
import tqdm
from pathlib import Path
from ..engine.engine import DPhysicsEngine
from ..engine.engine_state import PhysicsState
from ..utils.torch_utils import set_device
from ..utils.environment import make_x_y_grids
from ..configs import WorldConfig, PhysicsEngineConfig, RobotModelConfig
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = True

grid_res = 0.05  # 5cm per grid cell
max_coord = 6.4  # meters
compile_opts = {"max-autotune": True, "triton.cudagraphs": True}
save_loc = Path(__file__).parent.parent / "engine_benchmark_results"


def plot_and_save(results, device, compiled):
    fig = plt.figure(figsize=(12, 8), dpi=200)
    plt.plot(list(results.keys()), list(results.values()))
    plt.xlabel("Number of robots")
    plt.ylabel("Time per iteration (ms)")
    plt.title("Time per iteration vs. number of robots")
    plt.tight_layout(pad=1.0)
    plt.grid()
    min_time = int(min(results.values()))
    max_time = round(max(results.values())) + 1
    num_ticks = max_time - min_time  # spaced by 0.5ms
    plt.yticks(range(min_time, max_time, 2), [str(i) for i in range(min_time, max_time, 2)])
    plt.gca().set_xscale("log", base=2)
    plt.xticks([2**i for i in range(len(results))], [str(2**i) for i in range(len(results))])
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = save_loc / f"benchmark_{device}_{now}_{'compile' if compiled else 'eager'}.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")


def main(args):
    device = set_device(args.device, high_performance=args.hperf)
    robot_model = RobotModelConfig("marv")
    results = {}
    B = 1
    bar = tqdm.tqdm(total=args.max_num_robots)
    bar.update(1)
    while B <= args.max_num_robots:
        x_grid, y_grid = make_x_y_grids(max_coord, grid_res, B)
        x_grid = x_grid.to(device)
        y_grid = y_grid.to(device)
        z_grid = torch.zeros_like(x_grid)
        world_config = WorldConfig(x_grid, y_grid, z_grid, grid_res, max_coord)
        physics_engine_config = PhysicsEngineConfig(B)
        state = PhysicsState.dummy(batch_size=B, robot_model=robot_model).to(device)
        engine = DPhysicsEngine(physics_engine_config, robot_model, device)
        controls = torch.zeros((B, 2 * robot_model.num_joints), device=device)
        if args.compile:
            engine = torch.compile(engine, options=compile_opts)
            _ = engine(deepcopy(state), controls, world_config)
            bar.set_description_str(f"Compiled engine for {B} robots")
        s = time.perf_counter_ns()
        for i in range(args.num_runs):
            _ = engine(state, controls, world_config)
        e = time.perf_counter_ns()
        time_per_it_ms = (e - s) / (args.num_runs * 1e6)
        results[B] = time_per_it_ms
        bar.update(B)
        bar.set_postfix({"Time per iteration (ms)": time_per_it_ms})
        B *= 2
    bar.close()
    plot_and_save(results, device, args.compile)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hperf", action="store_true", default=False)
    p.add_argument("--num_runs", type=int, default=2000)
    p.add_argument("--max_num_robots", type=int, default=128)
    p.add_argument("--compile", action="store_true", default=False)
    args = p.parse_args()
    main(args)
