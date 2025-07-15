import os.path as osp
from pprint import pprint

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gymnasium as gym
import pathlib

from cfg_loader import load
from schedulers import RoundRobinScheduler, make_scheduler
from spark_sched_sim import metrics


ENV_CFG = {
    "num_executors": 10,
    "job_arrival_cap": 50,
    "job_arrival_rate": 4.0e-5,
    "moving_delay": 2000.0,
    "warmup_delay": 1000.0,
    "data_sampler_cls": "TPCHDataSampler",
    "render_mode": "human",
}


def main():
    # save final rendering to artifacts dir
    pathlib.Path("artifacts").mkdir(parents=True, exist_ok=True)

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--base_path",
        dest="base_path",
        help="base path to the models",
        required=True,
    )

    parser.add_argument(
        "--output_path",
        dest="output_path",
        help="path to the output file",
        required=True,
    )

    args = parser.parse_args()

    run_bulk_evaluation(args.base_path, args.output_path)


def run_bulk_evaluation(base_path, output_path):
    import json
    base_path = pathlib.Path(base_path)
    checkpoints_dir = base_path / "checkpoints"
    results = []

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoints_dir.exists() or not checkpoints_dir.is_dir():
        raise ValueError(f"Checkpoints directory not found: {checkpoints_dir}")

    # Only process checkpoint directories with numeric names
    numeric_checkpoints = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    for checkpoint in sorted(numeric_checkpoints, key=lambda x: int(x.name)):
        model_path = checkpoint / "model.pt"
        if model_path.exists():
            print(f"Evaluating checkpoint: {checkpoint.name}")
            avg_job_duration = decima(str(model_path))
            results.append({
                "checkpoint": checkpoint.name,
                "model_path": str(model_path),
                "avg_job_duration": avg_job_duration
            })
            # Incrementally write results after each checkpoint
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
        else:
            print(f"Skipping {checkpoint}: model.pt not found.")

    print(f"Results saved to {output_path}")


def decima(model_path):
    cfg = load(filename=osp.join("config", "decima_tpch.yaml"))

    agent_cfg = cfg["agent"] | {
        "num_executors": ENV_CFG["num_executors"],
        "state_dict_path": model_path,
    }

    scheduler = make_scheduler(agent_cfg)

    print("Example: Decima")
    print("Env settings:")
    pprint(ENV_CFG)

    print("Running episode...")
    avg_job_duration = run_episode(ENV_CFG, scheduler)

    print(f"Done! Average job duration: {avg_job_duration:.1f}s", flush=True)
    return avg_job_duration


def run_episode(env_cfg, scheduler, seed=1234):
    env = gym.make("spark_sched_sim:SparkSchedSimEnv-v0", env_cfg=env_cfg)

    if scheduler.env_wrapper_cls:
        env = scheduler.env_wrapper_cls(env)

    obs, _ = env.reset(seed=seed, options=None)
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = scheduler.schedule(obs)
        obs, _, terminated, truncated, _ = env.step(action)

    avg_job_duration = metrics.avg_job_duration(env) * 1e-3

    # cleanup rendering
    env.close()

    return avg_job_duration


if __name__ == "__main__":
    main()
