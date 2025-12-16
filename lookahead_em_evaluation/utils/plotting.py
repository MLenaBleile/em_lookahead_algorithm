"""
Plotting Utilities for EM Algorithm Evaluation

This module provides publication-quality visualization functions
for analyzing and comparing EM algorithm performance.

Example usage:
    >>> plot_convergence(
    ...     {'Standard EM': [likelihoods1], 'Lookahead': [likelihoods2]},
    ...     title='Convergence Comparison',
    ...     save_path='figures/convergence.png'
    ... )
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path


# =============================================================================
# Style Configuration
# =============================================================================

# Color palette (colorblind-friendly)
COLORS = {
    'standard_em': '#377eb8',      # Blue
    'lookahead_0.3': '#ff7f00',    # Orange
    'lookahead_0.5': '#4daf4a',    # Green
    'lookahead_0.7': '#f781bf',    # Pink
    'lookahead_0.9': '#a65628',    # Brown
    'lookahead_adaptive': '#e41a1c',  # Red
    'squarem': '#984ea3',          # Purple
    'quasi_newton': '#999999',     # Gray
}

# Default colors for algorithms not in the palette
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Line styles for distinguishing algorithms
LINE_STYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']

# Marker styles
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def get_color(algorithm: str, index: int = 0) -> str:
    """Get color for an algorithm."""
    # Normalize algorithm name
    algo_lower = algorithm.lower().replace(' ', '_').replace('-', '_')

    # Check for exact match
    if algo_lower in COLORS:
        return COLORS[algo_lower]

    # Check for partial match
    for key, color in COLORS.items():
        if key in algo_lower or algo_lower in key:
            return color

    # Default color
    return DEFAULT_COLORS[index % len(DEFAULT_COLORS)]


def save_figure(fig: Figure, save_path: Optional[str], dpi: int = 300) -> None:
    """
    Save figure to file if path provided.

    Args:
        fig: Matplotlib figure to save.
        save_path: Path to save to. Creates directory if needed.
        dpi: Resolution for raster formats.
    """
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')


# =============================================================================
# Main Plotting Functions
# =============================================================================

def plot_convergence(
    likelihood_histories: Dict[str, Union[List[float], List[List[float]]]],
    title: str = 'Log-Likelihood Convergence',
    xlabel: str = 'Iteration',
    ylabel: str = 'Log-Likelihood',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    show_std: bool = True,
    max_iter: Optional[int] = None
) -> Figure:
    """
    Plot convergence curves for multiple algorithms.

    Can handle both single runs (list of floats) and multiple runs
    (list of lists), showing mean ± std for the latter.

    Args:
        likelihood_histories: Dictionary mapping algorithm names to likelihood
            histories. Each history can be:
            - List[float]: Single run
            - List[List[float]]: Multiple runs (will show mean ± std)
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional path to save figure.
        figsize: Figure size (width, height) in inches.
        show_std: Whether to show ±1 std band for multiple runs.
        max_iter: Maximum iteration to plot (truncates if set).

    Returns:
        Matplotlib Figure object.

    Example:
        >>> histories = {
        ...     'Standard EM': [[-100, -90, -85], [-102, -91, -86]],
        ...     'Lookahead': [[-100, -88, -82], [-101, -89, -83]]
        ... }
        >>> fig = plot_convergence(histories, title='GMM Convergence')
        >>> plt.show()
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    for idx, (algo, histories) in enumerate(likelihood_histories.items()):
        color = get_color(algo, idx)
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]

        # Check if single run or multiple runs
        if len(histories) == 0:
            continue

        if isinstance(histories[0], (list, np.ndarray)):
            # Multiple runs - compute mean and std
            histories = [np.array(h) for h in histories]

            # Find min length (runs may have different lengths)
            min_len = min(len(h) for h in histories)
            if max_iter is not None:
                min_len = min(min_len, max_iter)

            # Truncate and stack
            truncated = np.array([h[:min_len] for h in histories])
            mean = np.mean(truncated, axis=0)
            std = np.std(truncated, axis=0)

            iterations = np.arange(min_len)

            ax.plot(iterations, mean, color=color, linestyle=linestyle,
                    label=algo, linewidth=2)

            if show_std and len(histories) > 1:
                ax.fill_between(iterations, mean - std, mean + std,
                               color=color, alpha=0.2)
        else:
            # Single run
            history = np.array(histories)
            if max_iter is not None:
                history = history[:max_iter]
            iterations = np.arange(len(history))

            ax.plot(iterations, history, color=color, linestyle=linestyle,
                    label=algo, linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_figure(fig, save_path)

    return fig


def plot_time_comparison(
    results_dict: Dict[str, List[float]],
    title: str = 'Time to Convergence',
    ylabel: str = 'Time (seconds)',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    show_means: bool = True
) -> Figure:
    """
    Create box plots comparing convergence times.

    Args:
        results_dict: Dictionary mapping algorithm names to lists of times.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Optional path to save figure.
        figsize: Figure size.
        show_means: Whether to show mean markers.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> times = {
        ...     'Standard EM': [45.2, 48.1, 43.5, 46.8],
        ...     'Lookahead': [32.1, 30.5, 33.2, 31.8]
        ... }
        >>> fig = plot_time_comparison(times)
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    algorithms = list(results_dict.keys())
    data = [results_dict[algo] for algo in algorithms]
    positions = np.arange(len(algorithms))

    # Create box plots
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)

    # Color the boxes
    for idx, (patch, algo) in enumerate(zip(bp['boxes'], algorithms)):
        color = get_color(algo, idx)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Show means
    if show_means:
        means = [np.mean(d) for d in data]
        ax.scatter(positions, means, marker='D', color='white',
                   s=50, zorder=5, edgecolors='black', linewidth=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    save_figure(fig, save_path)

    return fig


def plot_success_rates(
    results_dict: Dict[str, float],
    title: str = 'Success Rates by Algorithm',
    ylabel: str = 'Success Rate (%)',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    show_values: bool = True
) -> Figure:
    """
    Create bar chart of success rates.

    Args:
        results_dict: Dictionary mapping algorithm names to success rates.
                     Rates should be in [0, 1].
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Optional path to save figure.
        figsize: Figure size.
        show_values: Whether to show percentage labels on bars.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> rates = {'Standard EM': 0.85, 'Lookahead': 0.95, 'SQUAREM': 0.88}
        >>> fig = plot_success_rates(rates)
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    algorithms = list(results_dict.keys())
    rates = [results_dict[algo] * 100 for algo in algorithms]  # Convert to percentage
    positions = np.arange(len(algorithms))

    bars = ax.bar(positions, rates, width=0.6)

    # Color bars
    for idx, (bar, algo) in enumerate(zip(bars, algorithms)):
        color = get_color(algo, idx)
        bar.set_color(color)
        bar.set_alpha(0.8)

    # Add value labels
    if show_values:
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.annotate(f'{rate:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 105)  # Leave room for labels

    plt.tight_layout()
    save_figure(fig, save_path)

    return fig


def plot_memory_usage(
    results_dict: Dict[str, List[float]],
    title: str = 'Peak Memory Usage',
    ylabel: str = 'Memory (MB)',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> Figure:
    """
    Create bar chart of memory usage comparison.

    Args:
        results_dict: Dictionary mapping algorithm names to lists of
                     peak memory values (in MB).
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> memory = {
        ...     'Standard EM': [120, 125, 118],
        ...     'Lookahead': [145, 150, 142]
        ... }
        >>> fig = plot_memory_usage(memory)
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    algorithms = list(results_dict.keys())
    means = [np.mean(results_dict[algo]) for algo in algorithms]
    stds = [np.std(results_dict[algo]) for algo in algorithms]
    positions = np.arange(len(algorithms))

    bars = ax.bar(positions, means, width=0.6, yerr=stds, capsize=5)

    # Color bars
    for idx, (bar, algo) in enumerate(zip(bars, algorithms)):
        color = get_color(algo, idx)
        bar.set_color(color)
        bar.set_alpha(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    save_figure(fig, save_path)

    return fig


def plot_gamma_schedule(
    gamma_history: List[float],
    likelihood_history: List[float],
    title: str = 'Adaptive Gamma Schedule Effect',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5)
) -> Figure:
    """
    Plot gamma schedule alongside likelihood convergence.

    Creates two subplots:
    1. Gamma (γ) vs iteration
    2. Log-likelihood vs iteration

    Args:
        gamma_history: List of gamma values used at each iteration.
        likelihood_history: List of log-likelihood values.
        title: Overall figure title.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> gammas = [0.1, 0.15, 0.2, 0.25, 0.3, ...]
        >>> likelihoods = [-100, -95, -92, -90, ...]
        >>> fig = plot_gamma_schedule(gammas, likelihoods)
    """
    set_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    iterations = np.arange(len(gamma_history))

    # Plot gamma schedule
    ax1.plot(iterations, gamma_history, color='#e41a1c', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('γ (Gamma)')
    ax1.set_title('Gamma Schedule')
    ax1.set_ylim(0, 1.05)

    # Plot likelihood
    ax2.plot(iterations, likelihood_history, color='#377eb8', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Convergence')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, save_path)

    return fig


def plot_convergence_distribution(
    final_likelihoods: Dict[str, List[float]],
    true_likelihood: Optional[float] = None,
    title: str = 'Distribution of Final Likelihoods',
    xlabel: str = 'Log-Likelihood',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    bins: int = 20
) -> Figure:
    """
    Plot histogram of final likelihood values.

    Useful for analyzing local optima and comparing robustness.

    Args:
        final_likelihoods: Dict mapping algorithm names to lists of
                          final likelihood values from multiple runs.
        true_likelihood: Optional true/optimal likelihood for reference line.
        title: Plot title.
        xlabel: X-axis label.
        save_path: Optional path to save figure.
        figsize: Figure size.
        bins: Number of histogram bins.

    Returns:
        Matplotlib Figure object.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    for idx, (algo, values) in enumerate(final_likelihoods.items()):
        color = get_color(algo, idx)
        ax.hist(values, bins=bins, alpha=0.6, color=color,
                label=algo, edgecolor='white', linewidth=0.5)

    if true_likelihood is not None:
        ax.axvline(true_likelihood, color='red', linestyle='--',
                   linewidth=2, label='True Optimum')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, save_path)

    return fig


def plot_iteration_comparison(
    results_dict: Dict[str, List[int]],
    title: str = 'Iterations to Convergence',
    ylabel: str = 'Iterations',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> Figure:
    """
    Create box plots comparing number of iterations.

    Args:
        results_dict: Dict mapping algorithm names to lists of iteration counts.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    return plot_time_comparison(
        results_dict,
        title=title,
        ylabel=ylabel,
        save_path=save_path,
        figsize=figsize
    )


def plot_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = 'Performance Heatmap',
    cmap: str = 'RdYlGn',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    annotate: bool = True,
    fmt: str = '.2f'
) -> Figure:
    """
    Create a heatmap visualization.

    Useful for comparing algorithms across multiple tests or
    initialization strategies.

    Args:
        data: 2D array of values (rows x cols).
        row_labels: Labels for rows (e.g., test names).
        col_labels: Labels for columns (e.g., algorithm names).
        title: Plot title.
        cmap: Colormap name.
        save_path: Optional path to save figure.
        figsize: Figure size.
        annotate: Whether to show values in cells.
        fmt: Format string for annotations.

    Returns:
        Matplotlib Figure object.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap=cmap, aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)

    # Add annotations
    if annotate:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                value = data[i, j]
                text_color = 'white' if value < (data.max() + data.min()) / 2 else 'black'
                ax.text(j, i, format(value, fmt),
                       ha='center', va='center', color=text_color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)

    ax.set_title(title)

    plt.tight_layout()
    save_figure(fig, save_path)

    return fig


def create_summary_figure(
    convergence_data: Dict[str, List[List[float]]],
    time_data: Dict[str, List[float]],
    success_rates: Dict[str, float],
    title: str = 'Algorithm Comparison Summary',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 10)
) -> Figure:
    """
    Create a multi-panel summary figure.

    Combines convergence curves, time comparison, and success rates
    into a single publication-ready figure.

    Args:
        convergence_data: Dict of likelihood histories per algorithm.
        time_data: Dict of convergence times per algorithm.
        success_rates: Dict of success rates per algorithm.
        title: Overall figure title.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    set_publication_style()

    fig = plt.figure(figsize=figsize)

    # Create grid: 2 rows, 2 cols
    # Top row: convergence (spans both columns)
    # Bottom left: time comparison
    # Bottom right: success rates

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 4)

    algorithms = list(convergence_data.keys())

    # Plot 1: Convergence
    for idx, (algo, histories) in enumerate(convergence_data.items()):
        color = get_color(algo, idx)
        if len(histories) > 0:
            if isinstance(histories[0], (list, np.ndarray)):
                histories = [np.array(h) for h in histories]
                min_len = min(len(h) for h in histories)
                truncated = np.array([h[:min_len] for h in histories])
                mean = np.mean(truncated, axis=0)
                std = np.std(truncated, axis=0)
                iterations = np.arange(min_len)
                ax1.plot(iterations, mean, color=color, label=algo, linewidth=2)
                ax1.fill_between(iterations, mean - std, mean + std,
                                color=color, alpha=0.2)
            else:
                ax1.plot(np.arange(len(histories)), histories,
                        color=color, label=algo, linewidth=2)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Convergence Curves')
    ax1.legend(loc='lower right')

    # Plot 2: Time comparison (box plot)
    data = [time_data.get(algo, [0]) for algo in algorithms]
    positions = np.arange(len(algorithms))
    bp = ax2.boxplot(data, positions=positions, widths=0.6, patch_artist=True)

    for idx, (patch, algo) in enumerate(zip(bp['boxes'], algorithms)):
        color = get_color(algo, idx)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Time to Convergence')

    # Plot 3: Success rates (bar chart)
    rates = [success_rates.get(algo, 0) * 100 for algo in algorithms]
    bars = ax3.bar(positions, rates, width=0.6)

    for idx, (bar, algo) in enumerate(zip(bars, algorithms)):
        color = get_color(algo, idx)
        bar.set_color(color)
        bar.set_alpha(0.8)

    ax3.set_xticks(positions)
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rates')
    ax3.set_ylim(0, 105)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    save_figure(fig, save_path)

    return fig


# =============================================================================
# Unit Tests
# =============================================================================

def test_plot_convergence():
    """Test plot_convergence function."""
    print("Testing plot_convergence...")

    # Single runs
    histories = {
        'Standard EM': list(np.cumsum(-np.random.rand(50)) - 100),
        'Lookahead': list(np.cumsum(-np.random.rand(40) * 1.5) - 100)
    }

    fig = plot_convergence(histories, title='Test Convergence')
    assert fig is not None, "Should return a figure"
    plt.close(fig)

    # Multiple runs
    histories_multi = {
        'Standard EM': [list(np.cumsum(-np.random.rand(50)) - 100) for _ in range(5)],
        'Lookahead': [list(np.cumsum(-np.random.rand(40) * 1.5) - 100) for _ in range(5)]
    }

    fig = plot_convergence(histories_multi, title='Test Multi-Run')
    assert fig is not None, "Should return a figure"
    plt.close(fig)

    print("  PASSED")


def test_plot_time_comparison():
    """Test plot_time_comparison function."""
    print("Testing plot_time_comparison...")

    times = {
        'Standard EM': list(np.random.normal(45, 5, 20)),
        'Lookahead': list(np.random.normal(30, 3, 20)),
        'SQUAREM': list(np.random.normal(35, 4, 20))
    }

    fig = plot_time_comparison(times, title='Test Time Comparison')
    assert fig is not None, "Should return a figure"
    plt.close(fig)

    print("  PASSED")


def test_plot_success_rates():
    """Test plot_success_rates function."""
    print("Testing plot_success_rates...")

    rates = {
        'Standard EM': 0.85,
        'Lookahead': 0.95,
        'SQUAREM': 0.88
    }

    fig = plot_success_rates(rates, title='Test Success Rates')
    assert fig is not None, "Should return a figure"
    plt.close(fig)

    print("  PASSED")


def test_plot_memory_usage():
    """Test plot_memory_usage function."""
    print("Testing plot_memory_usage...")

    memory = {
        'Standard EM': list(np.random.normal(120, 10, 20)),
        'Lookahead': list(np.random.normal(145, 8, 20)),
        'SQUAREM': list(np.random.normal(180, 15, 20))
    }

    fig = plot_memory_usage(memory, title='Test Memory Usage')
    assert fig is not None, "Should return a figure"
    plt.close(fig)

    print("  PASSED")


def test_plot_gamma_schedule():
    """Test plot_gamma_schedule function."""
    print("Testing plot_gamma_schedule...")

    # Adaptive gamma schedule
    iterations = 50
    gammas = [min(0.9, 0.1 + 0.05 * t) for t in range(iterations)]
    likelihoods = list(np.cumsum(-np.random.rand(iterations)) - 100)

    fig = plot_gamma_schedule(gammas, likelihoods, title='Test Gamma Schedule')
    assert fig is not None, "Should return a figure"
    plt.close(fig)

    print("  PASSED")


def test_plot_heatmap():
    """Test plot_heatmap function."""
    print("Testing plot_heatmap...")

    data = np.random.rand(5, 4)
    rows = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5']
    cols = ['Standard EM', 'Lookahead', 'SQUAREM', 'Quasi-Newton']

    fig = plot_heatmap(data, rows, cols, title='Test Heatmap')
    assert fig is not None, "Should return a figure"
    plt.close(fig)

    print("  PASSED")


def test_save_figure():
    """Test that figures save correctly."""
    print("Testing save_figure...")

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'subdir', 'test.png')

        rates = {'A': 0.8, 'B': 0.9}
        fig = plot_success_rates(rates, save_path=save_path)
        plt.close(fig)

        assert os.path.exists(save_path), "Figure should be saved"
        assert os.path.getsize(save_path) > 1000, "Figure file should not be empty"

    print("  PASSED")


def test_create_summary_figure():
    """Test create_summary_figure function."""
    print("Testing create_summary_figure...")

    convergence = {
        'Standard EM': [list(np.cumsum(-np.random.rand(50)) - 100) for _ in range(3)],
        'Lookahead': [list(np.cumsum(-np.random.rand(40) * 1.5) - 100) for _ in range(3)]
    }

    times = {
        'Standard EM': list(np.random.normal(45, 5, 10)),
        'Lookahead': list(np.random.normal(30, 3, 10))
    }

    rates = {'Standard EM': 0.85, 'Lookahead': 0.95}

    fig = create_summary_figure(
        convergence, times, rates,
        title='Test Summary'
    )
    assert fig is not None, "Should return a figure"
    plt.close(fig)

    print("  PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running plotting.py unit tests")
    print("=" * 60)

    # Set non-interactive backend for testing
    import matplotlib
    matplotlib.use('Agg')

    test_plot_convergence()
    test_plot_time_comparison()
    test_plot_success_rates()
    test_plot_memory_usage()
    test_plot_gamma_schedule()
    test_plot_heatmap()
    test_save_figure()
    test_create_summary_figure()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
