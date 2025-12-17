"""
Resource Monitoring Utilities

This module provides tools for tracking execution time and memory usage
during algorithm runs.

Example usage:
    >>> monitor = ResourceMonitor()
    >>> monitor.start()
    >>> # ... run algorithm ...
    >>> stats = monitor.stop()
    >>> print(f"Time: {stats['elapsed_time']:.2f}s, Memory: {stats['peak_memory_mb']:.1f}MB")
"""

import time
import threading
import functools
from typing import Dict, Any, Callable, Optional

import psutil


class ResourceMonitor:
    """
    Monitor time and memory usage during algorithm execution.

    Tracks elapsed wall-clock time and peak memory usage using a background
    monitoring thread.

    Attributes:
        interval: Time between memory checks in seconds (default: 0.1)
        peak_memory: Peak memory usage observed in MB
        monitoring: Whether monitoring is currently active

    Example:
        >>> monitor = ResourceMonitor(interval=0.05)
        >>> monitor.start()
        >>> # Perform some memory-intensive computation
        >>> result = sum(range(10**7))
        >>> stats = monitor.stop()
        >>> print(f"Time: {stats['elapsed_time']:.2f}s")
        >>> print(f"Peak Memory: {stats['peak_memory_mb']:.1f}MB")
    """

    def __init__(self, interval: float = 0.1):
        """
        Initialize the resource monitor.

        Args:
            interval: Time between memory checks in seconds. Lower values
                     give more accurate peak memory tracking but use more CPU.
        """
        self.interval = interval
        self.peak_memory: float = 0.0
        self.monitoring: bool = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._process: Optional[psutil.Process] = None

    def start(self) -> None:
        """
        Start monitoring resources.

        Begins tracking elapsed time and spawns a background thread to
        monitor peak memory usage.

        Raises:
            RuntimeError: If monitoring is already active.
        """
        if self.monitoring:
            raise RuntimeError("Monitoring is already active. Call stop() first.")

        self.peak_memory = 0.0
        self.monitoring = True
        self._process = psutil.Process()

        # Record initial memory
        initial_mem = self._process.memory_info().rss / (1024 * 1024)
        self.peak_memory = initial_mem

        # Start background monitoring thread
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

        # Record start time after thread setup
        self._start_time = time.perf_counter()

    def stop(self) -> Dict[str, float]:
        """
        Stop monitoring and return collected statistics.

        Returns:
            Dictionary with:
                - 'elapsed_time': Wall-clock time in seconds
                - 'peak_memory_mb': Peak memory usage in megabytes

        Raises:
            RuntimeError: If monitoring was never started.
        """
        if not self.monitoring:
            raise RuntimeError("Monitoring was never started. Call start() first.")

        # Record end time first for accuracy
        end_time = time.perf_counter()

        # Stop monitoring thread
        self.monitoring = False
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)

        elapsed = end_time - self._start_time if self._start_time else 0.0

        return {
            'elapsed_time': elapsed,
            'peak_memory_mb': self.peak_memory
        }

    def _monitor(self) -> None:
        """
        Background monitoring loop.

        Continuously checks memory usage and updates peak_memory
        until monitoring is stopped.
        """
        while self.monitoring:
            try:
                if self._process is not None:
                    mem = self._process.memory_info().rss / (1024 * 1024)  # Convert to MB
                    self.peak_memory = max(self.peak_memory, mem)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process may have ended or access denied
                pass
            time.sleep(self.interval)

    def __enter__(self) -> 'ResourceMonitor':
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self.monitoring:
            self.stop()


def timed(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Wraps a function to measure and print its execution time.
    Also attaches timing info to the function's last_timing attribute.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function that prints timing info.

    Example:
        >>> @timed
        ... def slow_function(n):
        ...     return sum(range(n))
        >>> result = slow_function(10**6)
        slow_function took 0.05s
        >>> print(f"Last timing: {slow_function.last_timing:.4f}s")
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        wrapper.last_timing = elapsed
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result

    wrapper.last_timing = None
    return wrapper


def timed_silent(func: Callable) -> Callable:
    """
    Decorator to time function execution without printing.

    Like @timed but doesn't print. Access timing via func.last_timing.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function with timing info attached.

    Example:
        >>> @timed_silent
        ... def compute(n):
        ...     return sum(range(n))
        >>> result = compute(10**6)
        >>> print(f"Took: {compute.last_timing:.4f}s")
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        wrapper.last_timing = elapsed
        return result

    wrapper.last_timing = None
    return wrapper


def get_memory_usage() -> float:
    """
    Get current memory usage of this process in MB.

    Returns:
        Current memory usage in megabytes.

    Example:
        >>> mem_before = get_memory_usage()
        >>> big_list = list(range(10**6))
        >>> mem_after = get_memory_usage()
        >>> print(f"Used {mem_after - mem_before:.1f} MB")
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


# =============================================================================
# Unit Tests
# =============================================================================

def test_resource_monitor():
    """Test ResourceMonitor basic functionality."""
    print("Testing ResourceMonitor...")

    monitor = ResourceMonitor(interval=0.05)
    monitor.start()

    # Do some work that takes time and uses memory
    data = []
    for _ in range(100):
        data.append(list(range(10000)))
        time.sleep(0.01)

    stats = monitor.stop()

    # Verify we got reasonable values
    assert 'elapsed_time' in stats, "Missing elapsed_time"
    assert 'peak_memory_mb' in stats, "Missing peak_memory_mb"
    assert stats['elapsed_time'] > 0.5, f"Elapsed time too short: {stats['elapsed_time']}"
    assert stats['peak_memory_mb'] > 0, f"Peak memory should be positive: {stats['peak_memory_mb']}"

    print(f"  Elapsed time: {stats['elapsed_time']:.2f}s")
    print(f"  Peak memory: {stats['peak_memory_mb']:.1f}MB")
    print("  PASSED")


def test_resource_monitor_context_manager():
    """Test ResourceMonitor as context manager."""
    print("Testing ResourceMonitor context manager...")

    with ResourceMonitor(interval=0.05) as monitor:
        time.sleep(0.2)

    # After context manager, monitoring should be stopped
    assert not monitor.monitoring, "Monitoring should be stopped after context exit"
    print("  PASSED")


def test_timed_decorator():
    """Test @timed decorator."""
    print("Testing @timed decorator...")

    @timed
    def test_func(n):
        total = 0
        for i in range(n):
            total += i
        return total

    result = test_func(100000)

    assert result == sum(range(100000)), "Function result incorrect"
    assert test_func.last_timing is not None, "last_timing not set"
    assert test_func.last_timing > 0, "Timing should be positive"

    print(f"  Last timing: {test_func.last_timing:.6f}s")
    print("  PASSED")


def test_no_thread_leaks():
    """Test that monitoring thread stops properly."""
    print("Testing for thread leaks...")

    initial_threads = threading.active_count()

    for _ in range(5):
        monitor = ResourceMonitor(interval=0.02)
        monitor.start()
        time.sleep(0.05)
        monitor.stop()

    time.sleep(0.1)  # Give threads time to clean up
    final_threads = threading.active_count()

    assert final_threads <= initial_threads + 1, \
        f"Thread leak detected: {initial_threads} -> {final_threads}"

    print(f"  Initial threads: {initial_threads}")
    print(f"  Final threads: {final_threads}")
    print("  PASSED")


def test_memory_tracking():
    """Test that peak memory tracking works."""
    print("Testing memory tracking...")

    monitor = ResourceMonitor(interval=0.01)

    # Get baseline memory
    baseline = get_memory_usage()

    monitor.start()

    # Allocate significant memory
    big_data = [list(range(100000)) for _ in range(10)]
    time.sleep(0.1)  # Give monitor time to catch it

    stats = monitor.stop()

    # Peak should be higher than baseline
    assert stats['peak_memory_mb'] >= baseline, \
        f"Peak memory {stats['peak_memory_mb']:.1f} should be >= baseline {baseline:.1f}"

    print(f"  Baseline: {baseline:.1f}MB")
    print(f"  Peak: {stats['peak_memory_mb']:.1f}MB")
    print("  PASSED")

    # Clean up
    del big_data


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running timing.py unit tests")
    print("=" * 60)

    test_resource_monitor()
    test_resource_monitor_context_manager()
    test_timed_decorator()
    test_no_thread_leaks()
    test_memory_tracking()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
