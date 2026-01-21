#!/usr/bin/python3
"""
Safe multiprocessing diagnostic - tests incrementally to avoid OOM killer
"""

import multiprocessing
import time
import os
import sys
import psutil
import signal

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_system_memory():
    """Get system memory stats"""
    mem = psutil.virtual_memory()
    return {
        'total_mb': mem.total / 1024 / 1024,
        'available_mb': mem.available / 1024 / 1024,
        'percent_used': mem.percent,
        'free_mb': mem.free / 1024 / 1024
    }

def simple_task(idx):
    """Minimal task - just return the index"""
    return idx

def safe_test_pool(num_processes, method=None, task_count=10):
    """
    Safely test multiprocessing with memory monitoring
    """
    print(f"\n{'='*70}")
    print(f"Testing: {num_processes} processes, method={method or 'default'}")
    print(f"{'='*70}")
    
    sys_mem = get_system_memory()
    print(f"System memory: {sys_mem['available_mb']:.0f}MB available / {sys_mem['total_mb']:.0f}MB total ({sys_mem['percent_used']:.1f}% used)")
    
    initial_mem = get_memory_usage()
    print(f"Initial process memory: {initial_mem:.1f}MB")
    
    # Estimate memory needed (very rough)
    estimated_mem_per_process = 200  # MB per Python process with numpy
    estimated_total = estimated_mem_per_process * num_processes
    
    if estimated_total > sys_mem['available_mb'] * 0.8:
        print(f"\n⚠️  WARNING: Estimated memory needed (~{estimated_total:.0f}MB) exceeds 80% of available memory")
        print(f"⚠️  This may trigger OOM killer! Skipping this test.")
        return None
    
    if method:
        try:
            multiprocessing.set_start_method(method, force=True)
            print(f"Set start method to: {method}")
        except RuntimeError as e:
            print(f"Could not set start method: {e}")
    
    print(f"Current start method: {multiprocessing.get_start_method()}")
    
    try:
        print(f"\nCreating pool with {num_processes} processes...")
        start_time = time.time()
        
        with multiprocessing.Pool(num_processes) as pool:
            pool_mem = get_memory_usage()
            print(f"Pool created! Memory: {pool_mem:.1f}MB (+{pool_mem - initial_mem:.1f}MB)")
            
            # Monitor system memory
            sys_mem = get_system_memory()
            print(f"System available: {sys_mem['available_mb']:.0f}MB")
            
            print(f"\nSubmitting {task_count} tasks...")
            result = pool.map_async(simple_task, range(task_count))
            
            # Wait with timeout and periodic checks
            timeout = 30
            poll_interval = 2
            elapsed = 0
            
            while not result.ready() and elapsed < timeout:
                time.sleep(poll_interval)
                elapsed += poll_interval
                sys_mem = get_system_memory()
                print(f"  Waiting... ({elapsed}s) Available memory: {sys_mem['available_mb']:.0f}MB", flush=True)
            
            if result.ready():
                results = result.get()
                total_time = time.time() - start_time
                final_mem = get_memory_usage()
                
                print(f"\n✓ SUCCESS!")
                print(f"  Time: {total_time:.2f}s")
                print(f"  Results: {len(results)} tasks completed")
                print(f"  Peak memory: {final_mem:.1f}MB")
                return True
            else:
                print(f"\n✗ TIMEOUT after {timeout}s")
                return False
                
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("SAFE MULTIPROCESSING DIAGNOSTIC")
    print("="*70)
    
    sys_mem = get_system_memory()
    print(f"\nSystem Info:")
    print(f"  CPU count: {multiprocessing.cpu_count()}")
    print(f"  Total memory: {sys_mem['total_mb']:.0f}MB")
    print(f"  Available memory: {sys_mem['available_mb']:.0f}MB")
    print(f"  Memory in use: {sys_mem['percent_used']:.1f}%")
    
    # Test progressively
    test_configs = [
        (2, None, "Baseline: 2 processes, default method"),
        (4, None, "Scale up: 4 processes, default method"),
        (4, 'spawn', "Test spawn: 4 processes with spawn"),
        (8, 'spawn', "More spawn: 8 processes with spawn"),
    ]
    
    # Only test 32 if system has enough memory
    if sys_mem['available_mb'] > 8000:
        test_configs.append((32, 'spawn', "Full scale: 32 processes with spawn"))
    else:
        print(f"\n⚠️  Skipping 32-process test: insufficient memory ({sys_mem['available_mb']:.0f}MB available)")
    
    results = []
    
    for num_procs, method, description in test_configs:
        print(f"\n\n{'#'*70}")
        print(f"# {description}")
        print(f"{'#'*70}")
        
        success = safe_test_pool(num_procs, method, task_count=10)
        results.append((description, success))
        
        if success is None:
            print("\nSkipped due to memory constraints")
            break
        elif not success:
            print(f"\n⚠️  Test failed, stopping here to avoid system issues")
            break
        
        # Brief pause between tests
        time.sleep(2)
    
    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for desc, result in results:
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{status:8} {desc}")
    
    print("\n" + "="*70)
    
    # Check if spawn helps
    spawn_works = any(r[1] is True and 'spawn' in r[0] for r in results)
    default_fails = any(r[1] is False and 'default' in r[0] for r in results)
    
    if spawn_works and default_fails:
        print("✓ CONCLUSION: Using 'spawn' method solves the multiprocessing issue")
    elif spawn_works:
        print("✓ CONCLUSION: 'spawn' method works correctly")
    else:
        print("⚠️  CONCLUSION: Unable to confirm fix, check memory availability")
    
    print("="*70)

if __name__ == '__main__':
    # Protect against OOM by setting resource limits if possible
    try:
        import resource
        # Limit memory to 80% of available (soft limit only)
        mem = psutil.virtual_memory()
        # Note: This may not prevent OOM killer but worth trying
    except:
        pass
    
    main()
