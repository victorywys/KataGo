#!/usr/bin/python3
"""
Test script to verify multiprocessing deadlock issue and test the fix
"""

import multiprocessing
import time
import os

def simple_task(idx):
    """Simple task that prints and returns"""
    print(f"  Task {idx}: Starting", flush=True)
    time.sleep(0.1)
    print(f"  Task {idx}: Done", flush=True)
    return idx

def test_multiprocessing(num_processes, method=None):
    """Test multiprocessing with specified start method"""
    if method:
        try:
            multiprocessing.set_start_method(method, force=True)
            print(f"Set multiprocessing start method to: {method}")
        except RuntimeError:
            print(f"Could not set start method to {method}")
    
    print(f"Current start method: {multiprocessing.get_start_method()}")
    print(f"Testing with {num_processes} processes...\n")
    
    start = time.time()
    
    with multiprocessing.Pool(num_processes) as pool:
        print(f"Pool created, submitting 10 tasks...")
        result = pool.map_async(simple_task, range(10))
        
        # Wait with timeout
        try:
            results = result.get(timeout=30)
            elapsed = time.time() - start
            print(f"\n✓ SUCCESS: Completed in {elapsed:.2f}s")
            print(f"Results: {results}")
            return True
        except multiprocessing.TimeoutError:
            elapsed = time.time() - start
            print(f"\n✗ TIMEOUT: Deadlocked after {elapsed:.2f}s")
            return False

if __name__ == '__main__':
    print("="*70)
    print("Test 1: Default method with 32 processes (reproducing the issue)")
    print("="*70)
    success1 = test_multiprocessing(32)
    
    print("\n" + "="*70)
    print("Test 2: 'spawn' method with 32 processes (the fix)")
    print("="*70)
    success2 = test_multiprocessing(32, method='spawn')
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Default method: {'✓ PASSED' if success1 else '✗ FAILED (DEADLOCK)'}")
    print(f"Spawn method: {'✓ PASSED' if success2 else '✗ FAILED'}")
    
    if not success1 and success2:
        print("\n✓ Fix confirmed! Use 'spawn' method to avoid deadlock.")
