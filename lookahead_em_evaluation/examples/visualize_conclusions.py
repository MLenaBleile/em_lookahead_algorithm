"""
Visualization of Different Conclusions from EM vs Lookahead EM

This module creates visualizations showing how better likelihood
optimization can lead to fundamentally different conclusions about data.

Visualizations:
1. Expert assignment comparison (2D scatter with different colorings)
2. Model selection curves (BIC vs number of experts)
3. Expert boundary decision surfaces
4. Likelihood landscape comparison
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

# Add package root to path
pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_root not in sys.path:
    sys.path.insert(0, os.path.dirname(pkg_root))

from lookahead_em_evaluation.models.mixture_of_experts import MixtureOfExperts


def plot_expert_assignments(
    X: np.ndarray,
    y: np.ndarray,
    theta_std: Dict[str, np.ndarray],
    theta_la: Dict[str, np.ndarray],
    n_experts: int,
    dataset_name: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare expert assignments between standard and lookahead EM.

    Creates side-by-side scatter plots showing how each algorithm
    partitions the data into expert regions.
    """
    model = MixtureOfExperts(n_experts=n_experts, n_features=X.shape[1])

    # Get responsibilities and hard assignments
    resp_std = model.e_step((X, y), theta_std)
    resp_la = model.e_step((X, y), theta_la)

    assign_std = np.argmax(resp_std, axis=1)
    assign_la = np.argmax(resp_la, axis=1)

    # Use first 2 features for visualization
    x1, x2 = X[:, 0], X[:, 1] if X.shape[1] > 1 else X[:, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Standard EM assignments
    scatter1 = axes[0].scatter(x1, x2, c=assign_std, cmap='tab10', alpha=0.6, s=20)
    axes[0].set_title(f'Standard EM Assignments\n({len(np.unique(assign_std))} active experts)')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')

    # Lookahead EM assignments
    scatter2 = axes[1].scatter(x1, x2, c=assign_la, cmap='tab10', alpha=0.6, s=20)
    axes[1].set_title(f'Lookahead EM Assignments\n({len(np.unique(assign_la))} active experts)')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')

    # Differences
    different = assign_std != assign_la
    axes[2].scatter(x1[~different], x2[~different], c='gray', alpha=0.3, s=10, label='Same')
    axes[2].scatter(x1[different], x2[different], c='red', alpha=0.8, s=30, label='Different')
    axes[2].set_title(f'Assignment Differences\n({different.sum()} different, {100*different.mean():.1f}%)')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    axes[2].legend()

    fig.suptitle(f'{dataset_name}: Expert Assignment Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_model_selection(
    results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot BIC curves for model selection comparison.

    Shows how standard EM and lookahead EM may select different
    optimal numbers of experts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    expert_range = results['expert_range']

    # Log-likelihood comparison
    axes[0].plot(expert_range, results['standard_lls'], 'b-o', label='Standard EM', linewidth=2, markersize=8)
    axes[0].plot(expert_range, results['lookahead_lls'], 'r-s', label='Lookahead EM', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Experts', fontsize=12)
    axes[0].set_ylabel('Log-Likelihood', fontsize=12)
    axes[0].set_title('Log-Likelihood vs Model Complexity', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # BIC comparison
    axes[1].plot(expert_range, results['standard_bics'], 'b-o', label='Standard EM', linewidth=2, markersize=8)
    axes[1].plot(expert_range, results['lookahead_bics'], 'r-s', label='Lookahead EM', linewidth=2, markersize=8)

    # Mark optimal
    std_opt = results['standard_optimal_k']
    la_opt = results['lookahead_optimal_k']
    std_opt_bic = results['standard_bics'][expert_range.index(std_opt)]
    la_opt_bic = results['lookahead_bics'][expert_range.index(la_opt)]

    axes[1].axvline(std_opt, color='blue', linestyle='--', alpha=0.5)
    axes[1].axvline(la_opt, color='red', linestyle='--', alpha=0.5)
    axes[1].scatter([std_opt], [std_opt_bic], c='blue', s=150, zorder=5, marker='*', label=f'Std optimal (k={std_opt})')
    axes[1].scatter([la_opt], [la_opt_bic], c='red', s=150, zorder=5, marker='*', label=f'LA optimal (k={la_opt})')

    axes[1].set_xlabel('Number of Experts', fontsize=12)
    axes[1].set_ylabel('BIC (lower is better)', fontsize=12)
    axes[1].set_title('Model Selection via BIC', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Add annotation for different conclusions
    if results['different_conclusion']:
        fig.suptitle(f"{results['dataset']}: DIFFERENT CONCLUSIONS\n"
                     f"Standard EM selects {std_opt} experts, Lookahead EM selects {la_opt} experts",
                     fontsize=14, fontweight='bold', color='darkred')
    else:
        fig.suptitle(f"{results['dataset']}: Model Selection Comparison",
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_expert_slopes(
    theta_std: Dict[str, np.ndarray],
    theta_la: Dict[str, np.ndarray],
    feature_names: Optional[List[str]] = None,
    dataset_name: str = "Dataset",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare learned expert coefficients between methods.

    Shows how different optimization can lead to different
    interpretations of the data (different feature importance).
    """
    beta_std = theta_std['beta']
    beta_la = theta_la['beta']
    n_experts = beta_std.shape[0]
    n_features = beta_std.shape[1] - 1  # Exclude intercept

    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]

    fig, axes = plt.subplots(n_experts, 1, figsize=(10, 3 * n_experts))
    if n_experts == 1:
        axes = [axes]

    x = np.arange(n_features)
    width = 0.35

    for g in range(n_experts):
        axes[g].bar(x - width/2, beta_std[g, :-1], width, label='Standard EM', alpha=0.8)
        axes[g].bar(x + width/2, beta_la[g, :-1], width, label='Lookahead EM', alpha=0.8)
        axes[g].axhline(0, color='black', linewidth=0.5)
        axes[g].set_ylabel('Coefficient')
        axes[g].set_title(f'Expert {g+1}')
        axes[g].set_xticks(x)
        axes[g].set_xticklabels(feature_names, rotation=45, ha='right')
        if g == 0:
            axes[g].legend()

    fig.suptitle(f'{dataset_name}: Expert Regression Coefficients', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_gating_regions(
    X: np.ndarray,
    theta_std: Dict[str, np.ndarray],
    theta_la: Dict[str, np.ndarray],
    n_experts: int,
    dataset_name: str = "Dataset",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize gating function regions in 2D.

    Shows decision boundaries between expert regions.
    """
    model = MixtureOfExperts(n_experts=n_experts, n_features=X.shape[1])

    # Create grid
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 100),
        np.linspace(x2_min, x2_max, 100)
    )
    X_grid = np.column_stack([xx1.ravel(), xx2.ravel()])

    # Add dummy features if needed
    if X.shape[1] > 2:
        # Pad with zeros for extra features
        X_grid = np.column_stack([X_grid, np.zeros((len(X_grid), X.shape[1] - 2))])

    # Get gating probabilities on grid
    y_dummy = np.zeros(len(X_grid))

    # Standard EM gating
    X_aug = np.column_stack([X_grid, np.ones(len(X_grid))])
    logits_std = X_aug @ theta_std['gamma'].T
    probs_std = np.exp(logits_std - logits_std.max(axis=1, keepdims=True))
    probs_std /= probs_std.sum(axis=1, keepdims=True)
    assign_std = np.argmax(probs_std, axis=1).reshape(xx1.shape)

    # Lookahead EM gating
    logits_la = X_aug @ theta_la['gamma'].T
    probs_la = np.exp(logits_la - logits_la.max(axis=1, keepdims=True))
    probs_la /= probs_la.sum(axis=1, keepdims=True)
    assign_la = np.argmax(probs_la, axis=1).reshape(xx1.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Standard EM regions
    axes[0].contourf(xx1, xx2, assign_std, levels=n_experts, alpha=0.3, cmap='tab10')
    axes[0].contour(xx1, xx2, assign_std, levels=n_experts-1, colors='black', linewidths=0.5)
    axes[0].scatter(X[:, 0], X[:, 1], c='gray', s=5, alpha=0.5)
    axes[0].set_title('Standard EM Gating Regions')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')

    # Lookahead EM regions
    axes[1].contourf(xx1, xx2, assign_la, levels=n_experts, alpha=0.3, cmap='tab10')
    axes[1].contour(xx1, xx2, assign_la, levels=n_experts-1, colors='black', linewidths=0.5)
    axes[1].scatter(X[:, 0], X[:, 1], c='gray', s=5, alpha=0.5)
    axes[1].set_title('Lookahead EM Gating Regions')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')

    fig.suptitle(f'{dataset_name}: Expert Gating Regions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_summary_figure(
    results: List[Dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a summary figure comparing all experiments.
    """
    n_experiments = len(results)

    fig, axes = plt.subplots(n_experiments, 2, figsize=(12, 4 * n_experiments))
    if n_experiments == 1:
        axes = axes.reshape(1, -1)

    for i, r in enumerate(results):
        # Likelihood improvement
        names = ['Standard EM', 'Lookahead EM']
        lls = [r['standard_ll'], r['lookahead_ll']]
        colors = ['steelblue', 'coral']

        axes[i, 0].bar(names, lls, color=colors)
        axes[i, 0].set_ylabel('Log-Likelihood')
        axes[i, 0].set_title(f"{r['dataset_name']}: Log-Likelihood")
        improvement = r['ll_improvement_pct']
        axes[i, 0].annotate(f'+{improvement:.1f}%',
                            xy=(1, r['lookahead_ll']),
                            ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Expert utilization
        std_counts = r['standard_expert_counts']
        la_counts = r['lookahead_expert_counts']
        x = np.arange(len(std_counts))
        width = 0.35

        axes[i, 1].bar(x - width/2, std_counts, width, label='Standard EM', color='steelblue')
        axes[i, 1].bar(x + width/2, la_counts, width, label='Lookahead EM', color='coral')
        axes[i, 1].set_xlabel('Expert')
        axes[i, 1].set_ylabel('Sample Count')
        axes[i, 1].set_title(f"{r['dataset_name']}: Expert Utilization")
        axes[i, 1].legend()
        axes[i, 1].set_xticks(x)
        axes[i, 1].set_xticklabels([f'E{j+1}' for j in x])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# ============================================================
# Demonstration
# ============================================================

def run_visualization_demo(save_dir: Optional[str] = None):
    """
    Run all visualizations on sample data.
    """
    from real_datasets import (
        load_diabetes, load_california_housing,
        create_synthetic_multimodal, run_single_experiment,
        run_model_selection_experiment
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("VISUALIZATION DEMONSTRATION")
    print("=" * 60)

    # 1. Synthetic data (for clear visualization)
    print("\n1. Synthetic Data Visualization...")
    X, y, name = create_synthetic_multimodal(n_samples=500, n_features=2, n_true_experts=3, seed=42)

    result = run_single_experiment(
        X, y, name, n_experts=3,
        n_restarts=5, verbose=False
    )

    fig1 = plot_expert_assignments(
        X, y, result.standard_theta, result.lookahead_theta,
        n_experts=3, dataset_name=name,
        save_path=os.path.join(save_dir, 'expert_assignments.png') if save_dir else None
    )

    fig2 = plot_gating_regions(
        X, result.standard_theta, result.lookahead_theta,
        n_experts=3, dataset_name=name,
        save_path=os.path.join(save_dir, 'gating_regions.png') if save_dir else None
    )

    # 2. Model selection visualization
    print("\n2. Model Selection Visualization...")
    ms_result = run_model_selection_experiment(
        X, y, name,
        expert_range=range(2, 6),
        n_restarts=3,
        verbose=False
    )

    fig3 = plot_model_selection(
        ms_result,
        save_path=os.path.join(save_dir, 'model_selection.png') if save_dir else None
    )

    # 3. Expert slopes
    print("\n3. Expert Coefficients Visualization...")
    fig4 = plot_expert_slopes(
        result.standard_theta, result.lookahead_theta,
        feature_names=['X1', 'X2'],
        dataset_name=name,
        save_path=os.path.join(save_dir, 'expert_slopes.png') if save_dir else None
    )

    print("\nVisualization complete!")
    if save_dir:
        print(f"Figures saved to: {save_dir}")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize EM comparison results")
    parser.add_argument('--save-dir', type=str, default=None,
                        help="Directory to save figures")
    parser.add_argument('--no-show', action='store_true',
                        help="Don't display figures")

    args = parser.parse_args()

    run_visualization_demo(save_dir=args.save_dir)

    if not args.no_show:
        plt.show()
