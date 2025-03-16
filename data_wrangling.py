"""
GPU Utilities for Credit Card Fraud Detection
Helper functions for GPU acceleration
"""

import os
import time
import numpy as np

# Try to import GPU libraries
try:
    import cupy as cp
    import cudf
    import cuml
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("GPU libraries not available. Using CPU only.")

def check_gpu_availability():
    """Check if GPU is available and return status"""
    if not HAS_GPU:
        return False
    
    try:
        # Try to create a small array on GPU
        x = cp.array([1, 2, 3])
        return True
    except:
        return False

def check_gpu_memory():
    """Check available GPU memory"""
    if not HAS_GPU:
        return None
    
    try:
        mem_info = cp.cuda.runtime.memGetInfo()
        free_memory = mem_info[0] / 1024**3  # Convert to GB
        total_memory = mem_info[1] / 1024**3
        print(f"GPU Memory: {free_memory:.2f}GB free / {total_memory:.2f}GB total")
        return free_memory
    except:
        print("Could not check GPU memory")
        return None

def benchmark_function(func, *args, **kwargs):
    """Benchmark a function's execution time"""
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return result, execution_time

def compare_cpu_gpu_performance(func, *args, **kwargs):
    """Compare CPU vs GPU performance for a function"""
    # Run with CPU
    kwargs_cpu = kwargs.copy()
    kwargs_cpu['use_gpu'] = False
    cpu_result, cpu_time = benchmark_function(func, *args, **kwargs_cpu)
    
    # Run with GPU
    kwargs_gpu = kwargs.copy()
    kwargs_gpu['use_gpu'] = True
    gpu_result, gpu_time = benchmark_function(func, *args, **kwargs_gpu)
    
    # Calculate speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    
    print(f"CPU time: {cpu_time:.2f}s, GPU time: {gpu_time:.2f}s, Speedup: {speedup:.2f}x")
    return cpu_result, gpu_result, speedup

def load_data(file_path='transactions.txt', sample_size=None, use_gpu=False):
    """
    Load transaction data from a line-delimited JSON file with optional GPU acceleration.
    """
    print(f"Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return None
    
    try:
        # Check if GPU is requested and available
        gpu_available = False
        if use_gpu:
            try:
                import cudf
                gpu_available = True
            except ImportError:
                print("GPU libraries not available. Falling back to CPU for data loading.")
        
        if use_gpu and gpu_available:
            # GPU-accelerated loading
            df = cudf.read_json(file_path, lines=True)
            
            # Replace empty strings with NaN
            df = df.replace('', None)
            
            # Sample if needed
            if sample_size and len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)
                print(f"Sampled {len(df)} records for analysis.")
        else:
            # Original pandas loading
            df = pd.read_json(file_path, lines=True)
            
            # Replace empty strings with NaN
            df = df.replace('', np.nan)
            
            # Sample if needed
            if sample_size and len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)
                print(f"Sampled {len(df)} records for analysis.")
        
        print(f"Successfully loaded {len(df)} transactions with {len(df.columns)} columns.")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None