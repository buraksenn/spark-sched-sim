#!/usr/bin/env python3
"""
Enhanced plotting script for bulk evaluation results.
Generates multiple meaningful visualizations for thesis analysis.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(json_path: Path) -> Tuple[List[int], List[float]]:
    """Load and sort results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Ensure sorting by checkpoint (as int)
    data_sorted = sorted(data, key=lambda x: int(x['checkpoint']))
    checkpoints = [int(d['checkpoint']) for d in data_sorted]
    avg_durations = [d['avg_job_duration'] for d in data_sorted]
    return checkpoints, avg_durations

def load_all_results(outputs_dir: Path) -> Dict[str, Dict[str, Tuple[List[int], List[float]]]]:
    """Load all results organized by seed and model."""
    results = {}
    
    for seed_dir in outputs_dir.iterdir():
        if seed_dir.is_dir() and seed_dir.name.startswith('seed'):
            seed_name = seed_dir.name
            results[seed_name] = {}
            
            for json_file in seed_dir.glob('*.json'):
                model_name = json_file.stem
                try:
                    checkpoints, avg_durations = load_results(json_file)
                    results[seed_name][model_name] = (checkpoints, avg_durations)
                except Exception as e:
                    print(f"Warning: Could not load {json_file}: {e}")
    
    return results

def plot_learning_curves(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                        output_dir: Path, figsize=(15, 10)):
    """Plot learning curves for all models across seeds."""
    
    # Get all unique model names
    all_models = set()
    for seed_data in results.values():
        all_models.update(seed_data.keys())
    all_models = sorted(all_models)
    
    # Create subplots
    n_models = len(all_models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    for idx, model_name in enumerate(all_models):
        ax = axes[idx]
        
        for seed_name, seed_data in results.items():
            if model_name in seed_data:
                checkpoints, avg_durations = seed_data[model_name]
                ax.plot(checkpoints, avg_durations, marker='o', label=seed_name, linewidth=2, markersize=4)
        
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Average Job Duration (s)')
        ax.set_title(f'Model: {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(checkpoints) > 1:
            z = np.polyfit(checkpoints, avg_durations, 1)
            p = np.poly1d(z)
            ax.plot(checkpoints, p(checkpoints), "--", alpha=0.5, color='red', linewidth=1)
    
    # Hide empty subplots
    for idx in range(len(all_models), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to {output_dir / 'learning_curves.png'}")

def plot_convergence_analysis(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                            output_dir: Path, figsize=(12, 8)):
    """Plot convergence analysis showing final performance and improvement."""
    
    # Calculate final performance and improvement metrics
    final_performance = {}
    improvement_metrics = {}
    
    for seed_name, seed_data in results.items():
        for model_name, (checkpoints, avg_durations) in seed_data.items():
            if len(avg_durations) > 0:
                final_perf = avg_durations[-1]
                initial_perf = avg_durations[0]
                improvement = ((initial_perf - final_perf) / initial_perf) * 100
                
                key = f"{seed_name}_{model_name}"
                final_performance[key] = final_perf
                improvement_metrics[key] = improvement
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Final performance comparison
    keys = list(final_performance.keys())
    values = [final_performance[k] for k in keys]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(keys)]
    
    bars1 = ax1.bar(range(len(keys)), values, color=colors)
    ax1.set_xlabel('Seed-Model Combinations')
    ax1.set_ylabel('Final Average Job Duration (s)')
    ax1.set_title('Final Performance Comparison')
    ax1.set_xticks(range(len(keys)))
    ax1.set_xticklabels(keys, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Improvement percentage
    values = [improvement_metrics[k] for k in keys]
    bars2 = ax2.bar(range(len(keys)), values, color=colors)
    ax2.set_xlabel('Seed-Model Combinations')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Performance Improvement')
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(keys, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars2, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence analysis saved to {output_dir / 'convergence_analysis.png'}")

def plot_seed_comparison(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                        output_dir: Path, figsize=(14, 8)):
    """Plot comparison between different seeds for each model."""
    
    # Get all unique model names
    all_models = set()
    for seed_data in results.values():
        all_models.update(seed_data.keys())
    all_models = sorted(all_models)
    
    # Create subplots
    n_models = len(all_models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    for idx, model_name in enumerate(all_models):
        ax = axes[idx]
        
        # Collect data for this model across seeds
        seed_data = {}
        for seed_name, seed_results in results.items():
            if model_name in seed_results:
                checkpoints, avg_durations = seed_results[model_name]
                seed_data[seed_name] = (checkpoints, avg_durations)
        
        # Plot each seed
        for seed_name, (checkpoints, avg_durations) in seed_data.items():
            ax.plot(checkpoints, avg_durations, marker='o', label=seed_name, linewidth=2, markersize=4)
        
        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Average Job Duration (s)')
        ax.set_title(f'Seed Comparison: {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(all_models), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'seed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Seed comparison saved to {output_dir / 'seed_comparison.png'}")

def generate_summary_statistics(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                              output_dir: Path):
    """Generate summary statistics and save to CSV."""
    
    stats_data = []
    
    for seed_name, seed_data in results.items():
        for model_name, (checkpoints, avg_durations) in seed_data.items():
            if len(avg_durations) > 0:
                stats = {
                    'seed': seed_name,
                    'model': model_name,
                    'initial_performance': avg_durations[0],
                    'final_performance': avg_durations[-1],
                    'best_performance': min(avg_durations),
                    'worst_performance': max(avg_durations),
                    'improvement_percentage': ((avg_durations[0] - avg_durations[-1]) / avg_durations[0]) * 100,
                    'convergence_checkpoint': checkpoints[avg_durations.index(min(avg_durations))],
                    'total_checkpoints': len(checkpoints),
                    'std_deviation': np.std(avg_durations),
                    'mean_performance': np.mean(avg_durations)
                }
                stats_data.append(stats)
    
    # Create DataFrame and save
    df = pd.DataFrame(stats_data)
    df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"Summary statistics saved to {output_dir / 'summary_statistics.csv'}")
    
    # Print summary
    print("\n=== SUMMARY STATISTICS ===")
    print(df.round(2).to_string(index=False))
    
    return df

def plot_performance_heatmap(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                            output_dir: Path, figsize=(10, 6)):
    """Create a heatmap showing final performance across seeds and models."""
    
    # Prepare data for heatmap
    all_models = set()
    all_seeds = set()
    
    for seed_name, seed_data in results.items():
        all_seeds.add(seed_name)
        all_models.update(seed_data.keys())
    
    all_models = sorted(all_models)
    all_seeds = sorted(all_seeds)
    
    # Create performance matrix
    performance_matrix = np.full((len(all_seeds), len(all_models)), np.nan)
    
    for i, seed in enumerate(all_seeds):
        for j, model in enumerate(all_models):
            if seed in results and model in results[seed]:
                checkpoints, avg_durations = results[seed][model]
                if len(avg_durations) > 0:
                    performance_matrix[i, j] = avg_durations[-1]  # Final performance
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(performance_matrix, 
                xticklabels=all_models, 
                yticklabels=all_seeds,
                annot=True, 
                fmt='.1f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Average Job Duration (s)'})
    
    plt.title('Final Performance Heatmap')
    plt.xlabel('Models')
    plt.ylabel('Seeds')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance heatmap saved to {output_dir / 'performance_heatmap.png'}")

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive plots from bulk evaluation results.")
    parser.add_argument('outputs_dir', help='Path to outputs directory containing seed folders')
    parser.add_argument('--output_dir', '-o', default='plots', help='Directory to save plots (default: plots)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[15, 10], help='Figure size (width height)')
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    output_dir = Path(args.output_dir)
    
    if not outputs_dir.exists():
        print(f"Error: Outputs directory not found: {outputs_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    print("Loading results...")
    results = load_all_results(outputs_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded results for {len(results)} seeds")
    for seed_name, seed_data in results.items():
        print(f"  {seed_name}: {len(seed_data)} models")
    
    # Generate all plots
    print("\nGenerating plots...")
    
    # 1. Learning curves
    plot_learning_curves(results, output_dir, figsize=tuple(args.figsize))
    
    # 2. Convergence analysis
    plot_convergence_analysis(results, output_dir, figsize=tuple(args.figsize))
    
    # 3. Seed comparison
    plot_seed_comparison(results, output_dir, figsize=tuple(args.figsize))
    
    # 4. Performance heatmap
    plot_performance_heatmap(results, output_dir, figsize=tuple(args.figsize))
    
    # 5. Summary statistics
    generate_summary_statistics(results, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("Generated plots:")
    print("  - learning_curves.png: Learning curves for all models")
    print("  - convergence_analysis.png: Final performance and improvement comparison")
    print("  - seed_comparison.png: Seed comparison for each model")
    print("  - performance_heatmap.png: Performance heatmap across seeds and models")
    print("  - summary_statistics.csv: Detailed statistics")

if __name__ == "__main__":
    main() 