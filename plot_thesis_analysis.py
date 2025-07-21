#!/usr/bin/env python3
"""
Thesis-focused plotting script for RL-based Job Scheduling analysis.
Generates comprehensive academic visualizations with statistical rigor.
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
from scipy import stats
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

# Set style for academic publications
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_results(json_path: Path) -> Tuple[List[int], List[float]]:
    """Load and sort results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
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

def plot_individual_learning_curves(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                                   output_dir: Path):
    """Plot individual learning curves for each model (separate figures)."""
    
    all_models = set()
    for seed_data in results.values():
        all_models.update(seed_data.keys())
    
    for model_name in sorted(all_models):
        plt.figure(figsize=(12, 8))
        
        # Collect all data for this model across seeds
        all_checkpoints = []
        all_performances = []
        seed_colors = {}
        
        for i, (seed_name, seed_data) in enumerate(results.items()):
            if model_name in seed_data:
                checkpoints, avg_durations = seed_data[model_name]
                color = f'C{i}'
                seed_colors[seed_name] = color
                
                # Plot individual seed
                plt.plot(checkpoints, avg_durations, 'o-', color=color, 
                        label=f'{seed_name}', alpha=0.7, linewidth=2, markersize=4)
                
                # Collect for average
                all_checkpoints.extend(checkpoints)
                all_performances.extend(avg_durations)
        
        # Calculate and plot mean across seeds
        if len(all_checkpoints) > 0:
            # Group by checkpoint and calculate statistics
            checkpoint_stats = {}
            for cp, perf in zip(all_checkpoints, all_performances):
                if cp not in checkpoint_stats:
                    checkpoint_stats[cp] = []
                checkpoint_stats[cp].append(perf)
            
            # Calculate mean and std for each checkpoint
            sorted_checkpoints = sorted(checkpoint_stats.keys())
            means = [np.mean(checkpoint_stats[cp]) for cp in sorted_checkpoints]
            stds = [np.std(checkpoint_stats[cp]) for cp in sorted_checkpoints]
            
            # Plot mean with confidence interval
            plt.plot(sorted_checkpoints, means, 'k-', linewidth=3, label='Mean across seeds')
            plt.fill_between(sorted_checkpoints, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2, color='black', label='±1 std')
        
        plt.xlabel('Training Checkpoint')
        plt.ylabel('Average Job Duration (seconds)')
        plt.title(f'Learning Curve: {model_name.title()} Model\nRL-based Spark Job Scheduling')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add improvement annotation
        if len(means) > 1:
            initial_perf = means[0]
            final_perf = means[-1]
            improvement = ((initial_perf - final_perf) / initial_perf) * 100
            plt.text(0.05, 0.95, f'Average Improvement: {improvement:.1f}%', 
                    transform=plt.gca().transAxes, fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'learning_curve_{model_name}.png')
        plt.close()
        print(f"Individual learning curve saved: learning_curve_{model_name}.png")

def plot_convergence_analysis_detailed(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                                     output_dir: Path):
    """Detailed convergence analysis with statistical measures."""
    
    # Calculate convergence metrics
    convergence_data = []
    
    for seed_name, seed_data in results.items():
        for model_name, (checkpoints, avg_durations) in seed_data.items():
            if len(avg_durations) > 5:  # Need sufficient data
                # Find convergence point (where improvement becomes minimal)
                improvements = [avg_durations[i] - avg_durations[i+1] for i in range(len(avg_durations)-1)]
                convergence_threshold = np.std(improvements) * 0.5
                
                convergence_idx = len(improvements) - 1
                for i in range(len(improvements)-5):
                    if all(imp < convergence_threshold for imp in improvements[i:i+5]):
                        convergence_idx = i
                        break
                
                convergence_data.append({
                    'seed': seed_name,
                    'model': model_name,
                    'convergence_checkpoint': checkpoints[convergence_idx],
                    'convergence_performance': avg_durations[convergence_idx],
                    'final_performance': avg_durations[-1],
                    'initial_performance': avg_durations[0],
                    'improvement_rate': (avg_durations[0] - avg_durations[-1]) / len(checkpoints),
                    'stability': np.std(avg_durations[-10:]) if len(avg_durations) >= 10 else np.std(avg_durations)
                })
    
    df = pd.DataFrame(convergence_data)
    
    # Create subplots for detailed analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Convergence speed comparison
    sns.boxplot(data=df, x='model', y='convergence_checkpoint', ax=ax1)
    ax1.set_title('Convergence Speed Analysis')
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Convergence Checkpoint')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Final performance comparison with error bars
    performance_stats = df.groupby('model')['final_performance'].agg(['mean', 'std', 'count']).reset_index()
    
    bars = ax2.bar(performance_stats['model'], performance_stats['mean'], 
                   yerr=performance_stats['std'], capsize=5, alpha=0.7)
    ax2.set_title('Final Performance Comparison\n(Mean ± Standard Deviation)')
    ax2.set_xlabel('Model Type')
    ax2.set_ylabel('Average Job Duration (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, mean_val in zip(bars, performance_stats['mean']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{mean_val:.0f}s', ha='center', va='bottom', fontweight='bold')
    
    # 3. Learning efficiency (improvement rate)
    sns.boxplot(data=df, x='model', y='improvement_rate', ax=ax3)
    ax3.set_title('Learning Efficiency\n(Job Duration Reduction per Checkpoint)')
    ax3.set_xlabel('Model Type')
    ax3.set_ylabel('Improvement Rate (seconds/checkpoint)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Model stability analysis
    sns.boxplot(data=df, x='model', y='stability', ax=ax4)
    ax4.set_title('Model Stability Analysis\n(Lower is more stable)')
    ax4.set_xlabel('Model Type')
    ax4.set_ylabel('Performance Standard Deviation')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis_detailed.png')
    plt.close()
    print("Detailed convergence analysis saved: convergence_analysis_detailed.png")
    
    return df

def plot_performance_distributions(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                                 output_dir: Path):
    """Plot performance distributions for statistical analysis."""
    
    # Collect final performances
    performance_data = []
    
    for seed_name, seed_data in results.items():
        for model_name, (checkpoints, avg_durations) in seed_data.items():
            if len(avg_durations) > 0:
                performance_data.append({
                    'seed': seed_name,
                    'model': model_name,
                    'final_performance': avg_durations[-1],
                    'best_performance': min(avg_durations),
                    'initial_performance': avg_durations[0]
                })
    
    df = pd.DataFrame(performance_data)
    
    # Create violin plots for distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Final performance distributions
    sns.violinplot(data=df, x='model', y='final_performance', ax=ax1)
    sns.swarmplot(data=df, x='model', y='final_performance', ax=ax1, color='red', alpha=0.6, size=6)
    ax1.set_title('Final Performance Distributions\nRL-based Job Scheduling')
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Average Job Duration (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Best performance distributions
    sns.violinplot(data=df, x='model', y='best_performance', ax=ax2)
    sns.swarmplot(data=df, x='model', y='best_performance', ax=ax2, color='red', alpha=0.6, size=6)
    ax2.set_title('Best Performance Distributions\nAcross Training')
    ax2.set_xlabel('Model Type')
    ax2.set_ylabel('Average Job Duration (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_distributions.png')
    plt.close()
    print("Performance distributions saved: performance_distributions.png")

def plot_improvement_analysis(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                            output_dir: Path):
    """Analyze and plot improvement patterns."""
    
    improvement_data = []
    
    for seed_name, seed_data in results.items():
        for model_name, (checkpoints, avg_durations) in seed_data.items():
            if len(avg_durations) > 0:
                initial = avg_durations[0]
                final = avg_durations[-1]
                best = min(avg_durations)
                
                improvement_data.append({
                    'seed': seed_name,
                    'model': model_name,
                    'total_improvement': ((initial - final) / initial) * 100,
                    'potential_improvement': ((initial - best) / initial) * 100,
                    'efficiency': ((initial - final) / (initial - best)) * 100 if initial != best else 100
                })
    
    df = pd.DataFrame(improvement_data)
    
    # Create improvement analysis plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Total improvement comparison
    sns.barplot(data=df, x='model', y='total_improvement', ax=ax1, ci='sd')
    ax1.set_title('Total Performance Improvement\nby Model Type')
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Improvement Percentage (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add improvement values on bars
    for i, model in enumerate(df['model'].unique()):
        model_data = df[df['model'] == model]['total_improvement']
        mean_improvement = model_data.mean()
        ax1.text(i, mean_improvement + 1, f'{mean_improvement:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    # Learning efficiency vs potential
    ax2.scatter(df['potential_improvement'], df['total_improvement'], 
               c=[hash(model) for model in df['model']], alpha=0.7, s=100)
    
    # Add diagonal line for reference
    max_val = max(df['potential_improvement'].max(), df['total_improvement'].max())
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Learning')
    
    ax2.set_xlabel('Potential Improvement (%)')
    ax2.set_ylabel('Achieved Improvement (%)')
    ax2.set_title('Learning Efficiency Analysis\nAchieved vs Potential Improvement')
    ax2.legend()
    
    # Add model labels
    for i, row in df.iterrows():
        ax2.annotate(f"{row['seed']}-{row['model']}", 
                    (row['potential_improvement'], row['total_improvement']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_analysis.png')
    plt.close()
    print("Improvement analysis saved: improvement_analysis.png")

def plot_statistical_comparison(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                              output_dir: Path):
    """Statistical comparison with hypothesis testing."""
    
    # Prepare data for statistical tests
    model_performances = {}
    
    for seed_name, seed_data in results.items():
        for model_name, (checkpoints, avg_durations) in seed_data.items():
            if model_name not in model_performances:
                model_performances[model_name] = []
            model_performances[model_name].append(avg_durations[-1])  # Final performance
    
    # Perform pairwise t-tests
    models = list(model_performances.keys())
    n_models = len(models)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Statistical significance matrix
    p_values = np.ones((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                stat, p_val = stats.ttest_ind(model_performances[models[i]], 
                                            model_performances[models[j]])
                p_values[i][j] = p_val
    
    # Create heatmap of p-values
    sns.heatmap(p_values, annot=True, fmt='.3f', cmap='RdYlBu_r',
                xticklabels=models, yticklabels=models, ax=ax1,
                cbar_kws={'label': 'p-value'})
    ax1.set_title('Statistical Significance Matrix\n(p-values from t-tests)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)
    
    # Performance comparison with confidence intervals
    means = [np.mean(model_performances[model]) for model in models]
    stds = [np.std(model_performances[model]) for model in models]
    n_samples = [len(model_performances[model]) for model in models]
    
    # Calculate 95% confidence intervals
    confidence_intervals = [1.96 * std / np.sqrt(n) for std, n in zip(stds, n_samples)]
    
    bars = ax2.bar(models, means, yerr=confidence_intervals, capsize=5, alpha=0.7)
    ax2.set_title('Model Performance Comparison\n(Mean ± 95% Confidence Interval)')
    ax2.set_xlabel('Model Type')
    ax2.set_ylabel('Average Job Duration (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add significance indicators
    for i, (bar, mean_val, ci) in enumerate(zip(bars, means, confidence_intervals)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci + 10,
                f'{mean_val:.0f}±{ci:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_comparison.png')
    plt.close()
    print("Statistical comparison saved: statistical_comparison.png")

def plot_learning_dynamics(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                         output_dir: Path):
    """Plot learning dynamics with smoothed curves."""
    
    plt.figure(figsize=(14, 10))
    
    # Get all unique models
    all_models = set()
    for seed_data in results.values():
        all_models.update(seed_data.keys())
    all_models = sorted(all_models)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(all_models)]
    
    for model_idx, model_name in enumerate(all_models):
        all_data = []
        
        # Collect all training curves for this model
        for seed_name, seed_data in results.items():
            if model_name in seed_data:
                checkpoints, avg_durations = seed_data[model_name]
                
                # Normalize checkpoints to percentage of training
                max_checkpoint = max(checkpoints)
                normalized_checkpoints = [cp / max_checkpoint for cp in checkpoints]
                
                all_data.append((normalized_checkpoints, avg_durations))
        
        if all_data:
            # Create averaged learning curve
            # Interpolate all curves to common x-axis
            common_x = np.linspace(0, 1, 100)
            interpolated_curves = []
            
            for norm_cp, durations in all_data:
                if len(norm_cp) > 1:
                    interpolated = np.interp(common_x, norm_cp, durations)
                    interpolated_curves.append(interpolated)
            
            if interpolated_curves:
                mean_curve = np.mean(interpolated_curves, axis=0)
                std_curve = np.std(interpolated_curves, axis=0)
                
                # Plot mean with confidence band
                plt.plot(common_x * 100, mean_curve, color=colors[model_idx], 
                        linewidth=3, label=f'{model_name.title()}', alpha=0.8)
                plt.fill_between(common_x * 100, 
                               mean_curve - std_curve, 
                               mean_curve + std_curve,
                               color=colors[model_idx], alpha=0.2)
    
    plt.xlabel('Training Progress (%)')
    plt.ylabel('Average Job Duration (seconds)')
    plt.title('Learning Dynamics Comparison\nRL-based Spark Job Scheduling')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_dynamics.png')
    plt.close()
    print("Learning dynamics saved: learning_dynamics.png")

def generate_thesis_summary(results: Dict[str, Dict[str, Tuple[List[int], List[float]]]], 
                          output_dir: Path):
    """Generate comprehensive thesis summary with key findings."""
    
    # Calculate comprehensive statistics
    summary_stats = []
    
    for seed_name, seed_data in results.items():
        for model_name, (checkpoints, avg_durations) in seed_data.items():
            if len(avg_durations) > 0:
                initial = avg_durations[0]
                final = avg_durations[-1]
                best = min(avg_durations)
                worst = max(avg_durations)
                
                # Calculate learning metrics
                improvement_rate = (initial - final) / len(checkpoints)
                relative_improvement = ((initial - final) / initial) * 100
                convergence_efficiency = ((initial - final) / (initial - best)) * 100 if initial != best else 100
                
                summary_stats.append({
                    'seed': seed_name,
                    'model': model_name,
                    'initial_performance': initial,
                    'final_performance': final,
                    'best_performance': best,
                    'worst_performance': worst,
                    'relative_improvement_pct': relative_improvement,
                    'improvement_rate_per_checkpoint': improvement_rate,
                    'convergence_efficiency_pct': convergence_efficiency,
                    'performance_variance': np.var(avg_durations),
                    'training_checkpoints': len(checkpoints),
                    'best_checkpoint': checkpoints[avg_durations.index(best)]
                })
    
    df = pd.DataFrame(summary_stats)
    
    # Save detailed statistics
    df.to_csv(output_dir / 'thesis_summary_statistics.csv', index=False)
    
    # Generate summary by model
    model_summary = df.groupby('model').agg({
        'initial_performance': ['mean', 'std'],
        'final_performance': ['mean', 'std'], 
        'best_performance': ['mean', 'std'],
        'relative_improvement_pct': ['mean', 'std'],
        'improvement_rate_per_checkpoint': ['mean', 'std'],
        'convergence_efficiency_pct': ['mean', 'std'],
        'performance_variance': ['mean', 'std']
    }).round(2)
    
    model_summary.to_csv(output_dir / 'thesis_model_comparison.csv')
    
    # Create summary visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Model ranking by final performance
    final_perf_avg = df.groupby('model')['final_performance'].mean().sort_values()
    ax1.barh(range(len(final_perf_avg)), final_perf_avg.values, color='lightblue')
    ax1.set_yticks(range(len(final_perf_avg)))
    ax1.set_yticklabels(final_perf_avg.index)
    ax1.set_xlabel('Average Job Duration (seconds)')
    ax1.set_title('Model Ranking by Final Performance\n(Lower is Better)')
    
    # Add value labels
    for i, v in enumerate(final_perf_avg.values):
        ax1.text(v + 5, i, f'{v:.0f}s', va='center', fontweight='bold')
    
    # Improvement comparison
    improvement_avg = df.groupby('model')['relative_improvement_pct'].mean().sort_values(ascending=False)
    bars = ax2.bar(range(len(improvement_avg)), improvement_avg.values, color='lightgreen')
    ax2.set_xticks(range(len(improvement_avg)))
    ax2.set_xticklabels(improvement_avg.index, rotation=45)
    ax2.set_ylabel('Improvement Percentage (%)')
    ax2.set_title('Learning Effectiveness\n(Performance Improvement)')
    
    # Add value labels
    for bar, value in zip(bars, improvement_avg.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Learning efficiency
    efficiency_avg = df.groupby('model')['convergence_efficiency_pct'].mean().sort_values(ascending=False)
    ax3.bar(range(len(efficiency_avg)), efficiency_avg.values, color='orange', alpha=0.7)
    ax3.set_xticks(range(len(efficiency_avg)))
    ax3.set_xticklabels(efficiency_avg.index, rotation=45)
    ax3.set_ylabel('Efficiency Percentage (%)')
    ax3.set_title('Learning Efficiency\n(Achieved vs Potential Improvement)')
    
    # Stability analysis
    stability = 1 / (df.groupby('model')['performance_variance'].mean() + 1)  # Inverse of variance
    ax4.bar(range(len(stability)), stability.values, color='purple', alpha=0.7)
    ax4.set_xticks(range(len(stability)))
    ax4.set_xticklabels(stability.index, rotation=45)
    ax4.set_ylabel('Stability Score')
    ax4.set_title('Model Stability\n(Higher is More Stable)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'thesis_summary_analysis.png')
    plt.close()
    
    print("Thesis summary analysis saved: thesis_summary_analysis.png")
    print("Detailed statistics saved: thesis_summary_statistics.csv")
    print("Model comparison saved: thesis_model_comparison.csv")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive thesis plots for RL job scheduling.")
    parser.add_argument('outputs_dir', help='Path to outputs directory containing seed folders')
    parser.add_argument('--output_dir', '-o', default='thesis_plots', help='Directory to save plots')
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    output_dir = Path(args.output_dir)
    
    if not outputs_dir.exists():
        print(f"Error: Outputs directory not found: {outputs_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_all_results(outputs_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded results for {len(results)} seeds")
    for seed_name, seed_data in results.items():
        print(f"  {seed_name}: {len(seed_data)} models")
    
    print("\nGenerating thesis-focused plots...")
    
    # Generate all thesis plots
    plot_individual_learning_curves(results, output_dir)
    convergence_df = plot_convergence_analysis_detailed(results, output_dir)
    plot_performance_distributions(results, output_dir)
    plot_improvement_analysis(results, output_dir)
    plot_statistical_comparison(results, output_dir)
    plot_learning_dynamics(results, output_dir)
    summary_df = generate_thesis_summary(results, output_dir)
    
    print(f"\nAll thesis plots saved to: {output_dir}")
    print("\n=== GENERATED THESIS VISUALIZATIONS ===")
    print("Individual Learning Curves:")
    for model in sorted(set([m for seed_data in results.values() for m in seed_data.keys()])):
        print(f"  - learning_curve_{model}.png")
    print("\nComparative Analysis:")
    print("  - convergence_analysis_detailed.png")
    print("  - performance_distributions.png") 
    print("  - improvement_analysis.png")
    print("  - statistical_comparison.png")
    print("  - learning_dynamics.png")
    print("  - thesis_summary_analysis.png")
    print("\nStatistical Data:")
    print("  - thesis_summary_statistics.csv")
    print("  - thesis_model_comparison.csv")

if __name__ == "__main__":
    main() 