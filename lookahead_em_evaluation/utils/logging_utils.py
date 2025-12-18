"""
Experiment Logging Utilities

This module provides tools for logging experiment results to disk
in a structured, queryable format.

Example usage:
    >>> logger = ExperimentLogger(base_dir='results')
    >>> logger.log_run({
    ...     'test_id': 'test_1',
    ...     'algorithm': 'standard_em',
    ...     'seed': 42,
    ...     'final_likelihood': -2341.2,
    ...     # ... more fields
    ... })
    >>> results = logger.load_results('test_1', 'standard_em')
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy types.

    Converts numpy arrays to lists and numpy scalars to Python types.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ExperimentLogger:
    """
    Logger for saving and loading experiment results.

    Organizes results by test_id and algorithm, with automatic
    timestamping and unique run IDs.

    Attributes:
        base_dir: Root directory for all results.

    Directory structure:
        results/
        ├── test_1/
        │   ├── standard_em.jsonl
        │   ├── lookahead_adaptive.jsonl
        │   └── squarem.jsonl
        ├── test_2/
        │   └── ...
        └── summary.json

    Example:
        >>> logger = ExperimentLogger(base_dir='results')
        >>> logger.log_run({
        ...     'test_id': 'test_1',
        ...     'algorithm': 'standard_em',
        ...     'seed': 42,
        ...     'final_likelihood': -2341.2
        ... })
        >>> results = logger.load_results('test_1', 'standard_em')
        >>> print(f"Loaded {len(results)} runs")
    """

    def __init__(self, base_dir: str = 'results'):
        """
        Initialize the experiment logger.

        Args:
            base_dir: Root directory for storing results.
        """
        self.base_dir = Path(base_dir)
        self._ensure_base_dir()

    def _ensure_base_dir(self) -> None:
        """Create base directory if it doesn't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_test_dir(self, test_id: str) -> Path:
        """Get or create directory for a specific test."""
        test_dir = self.base_dir / test_id
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    def _get_results_file(self, test_id: str, algorithm: str) -> Path:
        """Get path to results file for a test/algorithm combination."""
        test_dir = self._get_test_dir(test_id)
        # Sanitize algorithm name for filename
        safe_name = algorithm.replace(' ', '_').replace('/', '_')
        return test_dir / f"{safe_name}.jsonl"

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        return str(uuid.uuid4())[:8]

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()

    def log_run(self, run_result: Dict[str, Any]) -> str:
        """
        Save a single run result to disk.

        Automatically adds timestamp and unique run_id if not present.

        Args:
            run_result: Dictionary containing run results.
                       Must include 'test_id' and 'algorithm' keys.

        Returns:
            The run_id assigned to this run.

        Raises:
            ValueError: If 'test_id' or 'algorithm' missing from run_result.

        Example:
            >>> run_id = logger.log_run({
            ...     'test_id': 'test_1',
            ...     'algorithm': 'standard_em',
            ...     'seed': 42,
            ...     'final_likelihood': -2341.2,
            ...     'iterations': 157,
            ...     'time_seconds': 45.2,
            ...     'converged': True
            ... })
            >>> print(f"Saved run: {run_id}")
        """
        if 'test_id' not in run_result:
            raise ValueError("run_result must contain 'test_id'")
        if 'algorithm' not in run_result:
            raise ValueError("run_result must contain 'algorithm'")

        # Add metadata if not present
        result = run_result.copy()
        if 'run_id' not in result:
            result['run_id'] = self._generate_run_id()
        if 'timestamp' not in result:
            result['timestamp'] = self._get_timestamp()

        # Get file path
        file_path = self._get_results_file(result['test_id'], result['algorithm'])

        # Append to JSONL file (one JSON object per line)
        with open(file_path, 'a') as f:
            json.dump(result, f, cls=NumpyEncoder)
            f.write('\n')

        return result['run_id']

    def load_results(
        self,
        test_id: str,
        algorithm: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load results from disk.

        Args:
            test_id: ID of the test to load.
            algorithm: Specific algorithm to load. If None, loads all algorithms.

        Returns:
            List of run result dictionaries.

        Example:
            >>> results = logger.load_results('test_1', 'standard_em')
            >>> print(f"Loaded {len(results)} runs")
            >>> for r in results:
            ...     print(f"  Seed {r['seed']}: {r['final_likelihood']:.1f}")
        """
        test_dir = self.base_dir / test_id

        if not test_dir.exists():
            return []

        results = []

        if algorithm is not None:
            # Load specific algorithm
            file_path = self._get_results_file(test_id, algorithm)
            if file_path.exists():
                results.extend(self._load_jsonl(file_path))
        else:
            # Load all algorithms
            for file_path in test_dir.glob('*.jsonl'):
                results.extend(self._load_jsonl(file_path))

        return results

    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load results from a JSONL file."""
        results = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        pass
        return results

    def list_algorithms(self, test_id: str) -> List[str]:
        """
        List all algorithms with results for a test.

        Args:
            test_id: ID of the test.

        Returns:
            List of algorithm names.
        """
        test_dir = self.base_dir / test_id

        if not test_dir.exists():
            return []

        algorithms = []
        for file_path in test_dir.glob('*.jsonl'):
            algorithms.append(file_path.stem)

        return sorted(algorithms)

    def load_all(self) -> List[Dict[str, Any]]:
        """
        Load all results from all JSONL files in the base directory.

        This is useful when the logger is pointed directly at a test results
        directory (e.g., results/test_1_high_d_gmm/) rather than the parent
        results directory.

        Returns:
            List of all run result dictionaries.
        """
        results = []

        # Load from all .jsonl files in base_dir
        for file_path in self.base_dir.glob('*.jsonl'):
            results.extend(self._load_jsonl(file_path))

        return results

    def list_tests(self) -> List[str]:
        """
        List all test IDs with saved results.

        Returns:
            List of test IDs.
        """
        tests = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                tests.append(item.name)
        return sorted(tests)

    def summary_stats(self, test_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Compute summary statistics across all algorithms for a test.

        Args:
            test_id: ID of the test.

        Returns:
            Dictionary mapping algorithm names to summary statistics.

        Example:
            >>> stats = logger.summary_stats('test_1')
            >>> for algo, s in stats.items():
            ...     print(f"{algo}: {s['n_runs']} runs, mean time {s['mean_time']:.1f}s")
        """
        from .metrics import compute_aggregate_statistics

        algorithms = self.list_algorithms(test_id)
        summary = {}

        for algorithm in algorithms:
            results = self.load_results(test_id, algorithm)
            if results:
                summary[algorithm] = compute_aggregate_statistics(results)

        return summary

    def save_summary(self, test_id: str) -> None:
        """
        Save summary statistics to a JSON file.

        Args:
            test_id: ID of the test.
        """
        summary = self.summary_stats(test_id)
        summary_path = self.base_dir / test_id / 'summary.json'

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

    def clear_results(self, test_id: str, algorithm: Optional[str] = None) -> int:
        """
        Clear saved results.

        Args:
            test_id: ID of the test.
            algorithm: Specific algorithm to clear. If None, clears all for test.

        Returns:
            Number of files deleted.

        Example:
            >>> n_deleted = logger.clear_results('test_1', 'standard_em')
            >>> print(f"Deleted {n_deleted} files")
        """
        test_dir = self.base_dir / test_id

        if not test_dir.exists():
            return 0

        deleted = 0

        if algorithm is not None:
            file_path = self._get_results_file(test_id, algorithm)
            if file_path.exists():
                file_path.unlink()
                deleted = 1
        else:
            for file_path in test_dir.glob('*.jsonl'):
                file_path.unlink()
                deleted += 1
            # Remove summary file too
            summary_path = test_dir / 'summary.json'
            if summary_path.exists():
                summary_path.unlink()
                deleted += 1

        return deleted

    def get_run_count(self, test_id: str, algorithm: Optional[str] = None) -> int:
        """
        Get number of runs saved.

        Args:
            test_id: ID of the test.
            algorithm: Specific algorithm. If None, counts all.

        Returns:
            Number of runs.
        """
        return len(self.load_results(test_id, algorithm))

    def to_dataframe(self, test_id: str, algorithm: Optional[str] = None):
        """
        Load results as a pandas DataFrame.

        Args:
            test_id: ID of the test.
            algorithm: Specific algorithm. If None, loads all.

        Returns:
            pandas DataFrame with results.

        Note:
            Requires pandas to be installed.
        """
        import pandas as pd

        results = self.load_results(test_id, algorithm)

        if not results:
            return pd.DataFrame()

        # Flatten nested dicts and filter out large arrays
        flat_results = []
        for r in results:
            flat = {}
            for k, v in r.items():
                if isinstance(v, (list, np.ndarray)) and len(v) > 10:
                    # Skip large arrays (like history)
                    flat[f'{k}_len'] = len(v)
                elif isinstance(v, dict):
                    # Flatten nested dicts
                    for k2, v2 in v.items():
                        if not isinstance(v2, (list, np.ndarray)) or len(v2) <= 10:
                            flat[f'{k}_{k2}'] = v2
                else:
                    flat[k] = v
            flat_results.append(flat)

        return pd.DataFrame(flat_results)


# =============================================================================
# Unit Tests
# =============================================================================

def test_basic_logging():
    """Test basic log and load functionality."""
    print("Testing basic logging...")

    import tempfile
    import shutil

    # Use temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        logger = ExperimentLogger(base_dir=temp_dir)

        # Log a run
        run_id = logger.log_run({
            'test_id': 'test_1',
            'algorithm': 'standard_em',
            'seed': 42,
            'final_likelihood': -2341.2,
            'iterations': 157,
            'time_seconds': 45.2
        })

        assert run_id is not None, "Should return run_id"
        assert len(run_id) == 8, f"Run ID should be 8 chars, got {len(run_id)}"

        # Load results
        results = logger.load_results('test_1', 'standard_em')
        assert len(results) == 1, f"Should have 1 result, got {len(results)}"
        assert results[0]['seed'] == 42, "Seed mismatch"
        assert results[0]['final_likelihood'] == -2341.2, "Likelihood mismatch"

        print("  PASSED")

    finally:
        shutil.rmtree(temp_dir)


def test_multiple_runs():
    """Test logging multiple runs."""
    print("Testing multiple runs...")

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        logger = ExperimentLogger(base_dir=temp_dir)

        # Log multiple runs
        for seed in range(5):
            logger.log_run({
                'test_id': 'test_1',
                'algorithm': 'standard_em',
                'seed': seed,
                'final_likelihood': -2340.0 - seed * 0.1,
                'time_seconds': 40.0 + seed
            })

        # Load all
        results = logger.load_results('test_1', 'standard_em')
        assert len(results) == 5, f"Should have 5 results, got {len(results)}"

        # Check ordering
        seeds = [r['seed'] for r in results]
        assert seeds == list(range(5)), "Seeds should be in order logged"

        print("  PASSED")

    finally:
        shutil.rmtree(temp_dir)


def test_multiple_algorithms():
    """Test logging multiple algorithms."""
    print("Testing multiple algorithms...")

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        logger = ExperimentLogger(base_dir=temp_dir)

        # Log for different algorithms
        for algo in ['standard_em', 'lookahead_adaptive', 'squarem']:
            logger.log_run({
                'test_id': 'test_1',
                'algorithm': algo,
                'seed': 42,
                'final_likelihood': -2340.0
            })

        # List algorithms
        algos = logger.list_algorithms('test_1')
        assert len(algos) == 3, f"Should have 3 algorithms, got {len(algos)}"

        # Load specific algorithm
        results = logger.load_results('test_1', 'lookahead_adaptive')
        assert len(results) == 1, "Should have 1 result for lookahead"

        # Load all
        all_results = logger.load_results('test_1')
        assert len(all_results) == 3, f"Should have 3 total results, got {len(all_results)}"

        print("  PASSED")

    finally:
        shutil.rmtree(temp_dir)


def test_numpy_serialization():
    """Test that numpy arrays are serialized correctly."""
    print("Testing numpy serialization...")

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        logger = ExperimentLogger(base_dir=temp_dir)

        # Log with numpy arrays
        logger.log_run({
            'test_id': 'test_1',
            'algorithm': 'standard_em',
            'seed': 42,
            'final_likelihood': np.float64(-2341.2),
            'iterations': np.int64(157),
            'likelihood_history': np.array([-100.0, -90.0, -85.0]),
            'mu': np.array([[1.0, 2.0], [3.0, 4.0]])
        })

        results = logger.load_results('test_1', 'standard_em')
        assert len(results) == 1, "Should have 1 result"

        r = results[0]
        assert r['iterations'] == 157, "Int serialization failed"
        assert abs(r['final_likelihood'] - (-2341.2)) < 0.001, "Float serialization failed"
        assert r['likelihood_history'] == [-100.0, -90.0, -85.0], "Array serialization failed"

        print("  PASSED")

    finally:
        shutil.rmtree(temp_dir)


def test_summary_stats():
    """Test summary statistics computation."""
    print("Testing summary_stats...")

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        logger = ExperimentLogger(base_dir=temp_dir)

        # Log runs for standard_em
        for seed in range(5):
            logger.log_run({
                'test_id': 'test_1',
                'algorithm': 'standard_em',
                'seed': seed,
                'time': 40.0 + seed,
                'iterations': 100 + seed * 10,
                'final_likelihood': -2340.0 - seed * 0.1
            })

        # Get summary
        summary = logger.summary_stats('test_1')

        assert 'standard_em' in summary, "Should have standard_em in summary"
        assert summary['standard_em']['n_runs'] == 5, "Should have 5 runs"
        assert summary['standard_em']['mean_time'] == 42.0, "Mean time should be 42"

        print("  PASSED")

    finally:
        shutil.rmtree(temp_dir)


def test_clear_results():
    """Test clearing results."""
    print("Testing clear_results...")

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        logger = ExperimentLogger(base_dir=temp_dir)

        # Log some runs
        for algo in ['standard_em', 'lookahead']:
            logger.log_run({
                'test_id': 'test_1',
                'algorithm': algo,
                'seed': 42
            })

        # Clear specific algorithm
        deleted = logger.clear_results('test_1', 'standard_em')
        assert deleted == 1, f"Should delete 1 file, got {deleted}"

        # Check remaining
        algos = logger.list_algorithms('test_1')
        assert 'lookahead' in algos, "Lookahead should remain"
        assert 'standard_em' not in algos, "Standard EM should be gone"

        # Clear all
        logger.log_run({'test_id': 'test_1', 'algorithm': 'standard_em', 'seed': 42})
        deleted = logger.clear_results('test_1')
        assert deleted >= 2, f"Should delete at least 2 files, got {deleted}"

        print("  PASSED")

    finally:
        shutil.rmtree(temp_dir)


def test_list_tests():
    """Test listing all tests."""
    print("Testing list_tests...")

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        logger = ExperimentLogger(base_dir=temp_dir)

        # Log to different tests
        for test_id in ['test_1', 'test_2', 'test_3']:
            logger.log_run({
                'test_id': test_id,
                'algorithm': 'standard_em',
                'seed': 42
            })

        tests = logger.list_tests()
        assert tests == ['test_1', 'test_2', 'test_3'], f"Wrong tests: {tests}"

        print("  PASSED")

    finally:
        shutil.rmtree(temp_dir)


def test_no_data_loss():
    """Test that appending doesn't lose data."""
    print("Testing no data loss on multiple writes...")

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    try:
        logger = ExperimentLogger(base_dir=temp_dir)

        # Write in batches
        for batch in range(3):
            for i in range(10):
                logger.log_run({
                    'test_id': 'test_1',
                    'algorithm': 'standard_em',
                    'seed': batch * 10 + i,
                    'batch': batch
                })

        results = logger.load_results('test_1', 'standard_em')
        assert len(results) == 30, f"Should have 30 results, got {len(results)}"

        print("  PASSED")

    finally:
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running logging_utils.py unit tests")
    print("=" * 60)

    test_basic_logging()
    test_multiple_runs()
    test_multiple_algorithms()
    test_numpy_serialization()
    test_summary_stats()
    test_clear_results()
    test_list_tests()
    test_no_data_loss()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
