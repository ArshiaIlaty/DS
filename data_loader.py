"""
Data Loader Module for Credit Card Fraud Detection
Functions for loading and describing transaction data
"""

import pandas as pd
import numpy as np
import json
import os
from tqdm.notebook import tqdm  # For Jupyter Notebook progress bars

def load_data(file_path='transactions.txt', sample_size=None):
    """
    Load transaction data from a line-delimited JSON file without using tqdm.
    """
    import pandas as pd
    import numpy as np
    import os
    
    print(f"Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return None
    
    try:
        # Direct approach using pandas
        df = pd.read_json(file_path, lines=True)
        
        # Replace empty strings with NaN
        df.replace('', np.nan, inplace=True)
        
        # Sample if needed
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
            print(f"Sampled {len(df)} records for analysis.")
        
        print(f"Successfully loaded {len(df)} transactions with {len(df.columns)} columns.")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
def describe_data(df):
    """
    Provide descriptive statistics and structure of the transaction data.
    
    Parameters:
    - df: DataFrame containing transaction data
    """
    print("\n" + "="*80)
    print("DATA STRUCTURE AND SUMMARY STATISTICS")
    print("="*80)
    
    # Basic information
    print(f"\nDataset contains {df.shape[0]} records with {df.shape[1]} fields.")
    
    # Data types and null values
    dtype_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique()
    })
    
    print("\nColumn Information:")
    print(dtype_info)
    
    # Summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    print("\nSummary statistics for numeric columns:")
    print(df[numeric_cols].describe())
    
    # Boolean columns
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        print("\nBoolean columns summary:")
        for col in bool_cols:
            true_pct = (df[col].sum() / len(df) * 100).round(2)
            print(f"- {col}: {true_pct}% True, {100-true_pct}% False")
    
    # Most common values in categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    print("\nCategorical columns overview:")
    for col in cat_cols:
        if df[col].nunique() < 20:  # Only show full distribution for columns with few values
            print(f"\n{col} distribution:")
            value_counts = df[col].value_counts(dropna=False)
            value_df = pd.DataFrame({
                'Value': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(df) * 100).round(2)
            })
            print(value_df)
        else:
            print(f"\nTop 10 values for {col} (out of {df[col].nunique()} unique values):")
            value_counts = df[col].value_counts(dropna=False).head(10)
            value_df = pd.DataFrame({
                'Value': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(df) * 100).round(2)
            })
            print(value_df)
    
    # Missing values
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if len(missing_values) > 0:
        print("\nColumns with missing values:")
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing %': (missing_values.values / len(df) * 100).round(2)
        })
        print(missing_df)
    else:
        print("\nNo missing values found in the dataset.")
    
    return dtype_info

def clean_data(df):
    """
    Perform basic data cleaning operations.
    
    Parameters:
    - df: DataFrame containing transaction data
    
    Returns:
    - Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert date/time columns to datetime
    date_cols = ['transactionDateTime', 'accountOpenDate', 'dateOfLastAddressChange']
    for col in date_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Fill missing values
    # For numeric columns, fill with median
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # For categorical columns, fill with most common value
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        most_common = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(most_common)
    
    # Fix data types if needed
    # Ensure boolean columns are correctly typed
    bool_cols = ['cardPresent', 'expirationDateKeyInMatch', 'isFraud']
    for col in bool_cols:
        if col in df_clean.columns:
            # Convert to boolean if not already
            if df_clean[col].dtype != 'bool':
                df_clean[col] = df_clean[col].astype(bool)
    
    print(f"Data cleaning complete. Shape: {df_clean.shape}")
    return df_clean

def save_data(df, output_path='dataset/transactions.pkl', protocol=4):
    """
    Save DataFrame to pickle file.
    
    Parameters:
    - df: DataFrame to save
    - output_path: Path where to save the pickle file
    - protocol: Pickle protocol version
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to pickle
        pd.to_pickle(df, output_path, protocol=protocol)
        print(f"Data successfully saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def load_saved_data(input_path='dataset/transactions.pkl'):
    """
    Load DataFrame from pickle file.
    
    Parameters:
    - input_path: Path to the pickle file
    
    Returns:
    - DataFrame if successful, None otherwise
    """
    try:
        if not os.path.exists(input_path):
            print(f"Error: File {input_path} not found!")
            return None
            
        df = pd.read_pickle(input_path)
        print(f"Successfully loaded data from {input_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from pickle: {e}")
        return None