import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Ensure sorting by checkpoint (as int)
    data_sorted = sorted(data, key=lambda x: int(x['checkpoint']))
    checkpoints = [int(d['checkpoint']) for d in data_sorted]
    avg_durations = [d['avg_job_duration'] for d in data_sorted]
    return checkpoints, avg_durations


def main():
    parser = argparse.ArgumentParser(description="Plot avg_job_duration vs. checkpoint from one or more JSON results files.")
    parser.add_argument('json_files', nargs='+', help='Path(s) to JSON output file(s) from get_bulk_results.py')
    parser.add_argument('--output', '-o', default=None, help='Optional path to save the plot (e.g., plot.png)')
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))

    for json_file in args.json_files:
        checkpoints, avg_durations = load_results(json_file)
        label = Path(json_file).stem
        plt.plot(checkpoints, avg_durations, marker='o', label=label)

    plt.xlabel('Checkpoint')
    plt.ylabel('Average Job Duration (s)')
    plt.title('Average Job Duration vs. Checkpoint')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)
        print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main() 