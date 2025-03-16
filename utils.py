"""
Utilities Module for Credit Card Fraud Detection
Common functions used across multiple scripts
"""

import os
import pandas as pd
import numpy as np
import time

def ensure_directories(directories):
    """
    Create multiple directories if they don't exist.
    
    Parameters:
    - directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_standard_directories(base_dir='results'):
    """
    Create standard directory structure for fraud detection project.
    
    Parameters:
    - base_dir: Base directory for results
    """
    directories = [
        f'{base_dir}/data',
        f'{base_dir}/models',
        f'{base_dir}/plots',
        f'{base_dir}/metrics'
    ]
    ensure_directories(directories)
    return directories

def load_data_by_extension(file_path, sample_size=None):
    """
    Load data based on file extension.
    
    Parameters:
    - file_path: Path to the data file
    - sample_size: Number of records to sample (optional)
    
    Returns:
    - DataFrame with loaded data or None if error occurs
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found!")
            return None
            
        print(f"Loading data from {file_path}...")
        
        # Load based on file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        elif file_path.endswith('.pkl'):
            df = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Sample if needed
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
            print(f"Sampled {len(df)} records from the dataset")
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def time_function(func, *args, **kwargs):
    """
    Time the execution of a function.
    
    Parameters:
    - func: Function to time
    - *args, **kwargs: Arguments to pass to the function
    
    Returns:
    - Result of the function and execution time
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    print(f"Time taken: {execution_time:.2f} seconds")
    return result, execution_time

def clean_data_for_analysis(df):
    """
    Clean data for analysis by handling nulls and invalid values.
    
    Parameters:
    - df: DataFrame to clean
    
    Returns:
    - Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Replace infinite values with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Handle critical columns for duplicate analysis
    critical_cols = ['accountNumber', 'transactionDateTime', 'transactionAmount', 
                     'merchantName', 'transactionType']
    
    for col in critical_cols:
        if col in df_clean.columns:
            if col == 'accountNumber':
                df_clean[col] = df_clean[col].fillna(999999999)
            elif col == 'transactionDateTime':
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(pd.Timestamp('2016-01-01'))
            elif col == 'transactionAmount':
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif col == 'merchantName':
                df_clean[col] = df_clean[col].fillna('Unknown Merchant')
            elif col == 'transactionType':
                df_clean[col] = df_clean[col].fillna('PURCHASE')
    
    # Handle other columns
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col not in critical_cols:
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col not in critical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean 