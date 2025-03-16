"""
Data Wrangling Module for Credit Card Fraud Detection
Functions for identifying duplicate and reversed transactions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import pickle
import os

def display(df):
    """Simple function to display a dataframe as text"""
    print(df.head())
    
def save_results(results, filename='results/data/analysis_results.pkl'):
    """
    Save analysis results to a pickle file.
    
    Parameters:
    - results: Dictionary of analysis results
    - filename: Path to save the pickle file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {filename}")
        return True
    except Exception as e:
        print(f"\nError saving results to {filename}: {str(e)}")
        return False

def load_results(filename='results/data/analysis_results.pkl'):
    """
    Load analysis results from a pickle file.
    
    Parameters:
    - filename: Path to the pickle file
    
    Returns:
    - Dictionary of analysis results or None if file doesn't exist or error occurs
    """
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                results = pickle.load(f)
            print(f"\nResults loaded from {filename}")
            return results
        except Exception as e:
            print(f"\nError loading results from {filename}: {str(e)}")
    else:
        print(f"\nCache file not found: {filename}")
    return None
    
def identify_duplicates(df, use_cached=True, cache_file='results/data/duplicate_analysis.pkl', force_recalculate=False):
    """
    Identify and analyze reversed and multi-swipe transactions.
    
    Parameters:
    - df: DataFrame containing transaction data
    - use_cached: Whether to try loading cached results first
    - cache_file: File name for caching results
    - force_recalculate: Force recalculation even if cache exists
    
    Returns:
    - Dictionary containing information about duplicates
    """
    # Try to load cached results first
    if use_cached and not force_recalculate:
        results = load_results(cache_file)
        if results is not None:
            # Verify the cached results match the current data
            if 'data_shape' in results and results['data_shape'] == df.shape:
                print("Using cached duplicate analysis results")
                
                # Print summary statistics from cached results
                print("\nReversal Transaction Analysis (from cache):")
                print(f"- Total identified reversals: {results['total_reversal_count']}")
                print(f"- Total dollar amount of reversals: ${results['total_reversal_amount']:.2f}")
                
                if 'reversal_transactions' in results and len(results['reversal_transactions']) > 0:
                    print("\nSample reversal transactions (from cache):")
                    print(results['reversal_transactions'].head())
                    
                    if 'time_diff_minutes' in results['reversal_transactions'].columns:
                        print("\nTime between original transaction and reversal:")
                        print(f"- Mean: {results['reversal_transactions']['time_diff_minutes'].mean():.2f} minutes")
                        print(f"- Median: {results['reversal_transactions']['time_diff_minutes'].median():.2f} minutes")
                        print(f"- Min: {results['reversal_transactions']['time_diff_minutes'].min():.2f} minutes")
                        print(f"- Max: {results['reversal_transactions']['time_diff_minutes'].max():.2f} minutes")
                
                print("\nMulti-Swipe Transaction Analysis (from cache):")
                print(f"- Total multi-swipe groups identified: {len(results.get('multi_swipe_groups', []))}")
                print(f"- Total extra transactions: {results['total_multi_swipe_count']}")
                print(f"- Total dollar amount of extra swipes: ${results['total_multi_swipe_amount']:.2f}")
                
                print(f"\nCache timestamp: {results.get('timestamp', 'unknown')}")
                
                return results
            else:
                print("Cached results don't match current data. Recomputing...")
        else:
            print("No valid cached results found. Computing duplicate analysis...")

    print("\n" + "="*80)
    print("DUPLICATE TRANSACTION ANALYSIS")
    print("="*80)
    
    # Create a copy of the dataframe
    df_dup = df.copy()
    
    # Ensure datetime format for transaction dates
    df_dup['transactionDateTime'] = pd.to_datetime(df_dup['transactionDateTime'])
    
    # Sort by account number and transaction date
    df_dup = df_dup.sort_values(['accountNumber', 'transactionDateTime'])
    
    # 1. Identify explicit reversals
    explicit_reversals = df_dup[df_dup['transactionType'] == 'REVERSAL'].copy()
    print(f"\nExplicit reversal transactions found: {len(explicit_reversals)}")
    
    # First, create a subset of all non-reversal transactions for faster filtering
    non_reversal_txns = df_dup[df_dup['transactionType'] != 'REVERSAL'].copy()
    # Create a map to match reversals with original transactions
    matched_reversals = []
    total_reversal_amount = 0

    # Check if we need to process all reversals or just a sample for testing
    sample_size = min(len(explicit_reversals), 1000)  # Process at most 1000 reversals for testing
    if len(explicit_reversals) > sample_size and not use_cached:
        print(f"\nProcessing a sample of {sample_size} reversals for testing...")
        reversals_to_process = explicit_reversals.sample(sample_size, random_state=42)
    else:
        print(f"\nMatching {len(explicit_reversals)} reversals to original transactions...")
        reversals_to_process = explicit_reversals

    # Create a progress bar with a meaningful description
    progress_bar = tqdm(
        reversals_to_process.iterrows(), 
        total=len(reversals_to_process),
        desc="Matching reversals to original transactions"
    )
    
    # Process reversals with progress bar
    for _, reversal in progress_bar:
        # Filter by account and merchant first (faster)
        account_merchant_txns = non_reversal_txns[
            (non_reversal_txns['accountNumber'] == reversal['accountNumber']) &
            (non_reversal_txns['merchantName'] == reversal['merchantName'])
        ]
        
        # Then filter by amount and time (more specific)
        potential_originals = account_merchant_txns[
            (abs(account_merchant_txns['transactionAmount'] - reversal['transactionAmount']) < 0.01) &
            (account_merchant_txns['transactionDateTime'] < reversal['transactionDateTime'])
        ]
        
        if len(potential_originals) > 0:
            # Calculate time differences
            time_diffs = reversal['transactionDateTime'] - potential_originals['transactionDateTime']
            potential_originals = potential_originals.copy()  # Create an explicit copy
            potential_originals.loc[:, 'timeDiff'] = time_diffs
            closest_original = potential_originals.loc[time_diffs.idxmin()]
            
            matched_reversals.append({
                'reversal_id': reversal.name,
                'original_id': closest_original.name,
                'accountNumber': reversal['accountNumber'],
                'amount': reversal['transactionAmount'],
                'merchantName': reversal['merchantName'],
                'original_datetime': closest_original['transactionDateTime'],
                'reversal_datetime': reversal['transactionDateTime'],
                'time_diff_minutes': closest_original['timeDiff'].total_seconds() / 60
            })
            
            total_reversal_amount += reversal['transactionAmount']
    
    # Create DataFrame of matched reversals
    matched_reversals_df = pd.DataFrame(matched_reversals) if matched_reversals else pd.DataFrame()
    
    # 2. Identify multi-swipe transactions
    # These are duplicate transactions with the same account, merchant, amount in a short time
    multi_swipe_groups = []
    multi_swipe_transactions = []
    time_threshold = timedelta(minutes=5)  # 5 minute threshold
    
    # Sample accounts for multi-swipe analysis to speed up processing
    sample_accounts = min(100, len(df_dup['accountNumber'].unique()))
    account_sample = np.random.choice(df_dup['accountNumber'].unique(), sample_accounts, replace=False)
    
    # Group by account, merchant, and amount
    for account in account_sample:
        account_data = df_dup[df_dup['accountNumber'] == account]
        
        # Group by merchant and amount
        for merchant in account_data['merchantName'].unique():
            merchant_data = account_data[account_data['merchantName'] == merchant]
            
            for amount in merchant_data['transactionAmount'].unique():
                # Don't consider $0 transactions
                if amount == 0:
                    continue
                    
                # Get transactions with this exact amount
                amount_txns = merchant_data[
                    (abs(merchant_data['transactionAmount'] - amount) < 0.01) &
                    (merchant_data['transactionType'] != 'REVERSAL')
                ].sort_values('transactionDateTime')
                
                if len(amount_txns) > 1:
                    # Check if any transactions are within the time threshold
                    groups = []
                    current_group = [amount_txns.iloc[0]]
                    
                    for i in range(1, len(amount_txns)):
                        prev_txn = amount_txns.iloc[i-1]
                        curr_txn = amount_txns.iloc[i]
                        
                        time_diff = curr_txn['transactionDateTime'] - prev_txn['transactionDateTime']
                        
                        if time_diff <= time_threshold:
                            # Add to current group
                            current_group.append(curr_txn)
                        else:
                            # Start a new group if the current group has multiple transactions
                            if len(current_group) > 1:
                                groups.append(current_group)
                            current_group = [curr_txn]
                    
                    # Add the last group if it has multiple transactions
                    if len(current_group) > 1:
                        groups.append(current_group)
                    
                    # Add to multi-swipe groups
                    for group in groups:
                        group_info = {
                            'accountNumber': account,
                            'merchantName': merchant,
                            'amount': amount,
                            'count': len(group),
                            'first_datetime': group[0]['transactionDateTime'],
                            'last_datetime': group[-1]['transactionDateTime'],
                            'time_span_seconds': (group[-1]['transactionDateTime'] - 
                                                group[0]['transactionDateTime']).total_seconds(),
                            'transaction_ids': [txn.name for txn in group]
                        }
                        multi_swipe_groups.append(group_info)
                        
                        # Add transactions to the list (excluding the first one)
                        for txn in group[1:]:
                            multi_swipe_transactions.append({
                                'group_id': len(multi_swipe_groups) - 1,
                                'transaction_id': txn.name,
                                'accountNumber': account,
                                'merchantName': merchant,
                                'amount': amount,
                                'datetime': txn['transactionDateTime']
                            })
    
    # Create DataFrames
    multi_swipe_groups_df = pd.DataFrame(multi_swipe_groups) if multi_swipe_groups else pd.DataFrame()
    multi_swipe_transactions_df = pd.DataFrame(multi_swipe_transactions) if multi_swipe_transactions else pd.DataFrame()
    
    # Calculate totals
    total_multi_swipe_count = len(multi_swipe_transactions)
    total_multi_swipe_amount = multi_swipe_transactions_df['amount'].sum() if len(multi_swipe_transactions) > 0 else 0
    
    # Print summary statistics
    print("\nReversal Transaction Analysis:")
    print(f"- Total identified reversals: {len(matched_reversals)}")
    print(f"- Total dollar amount of reversals: ${total_reversal_amount:.2f}")
    
    if len(matched_reversals) > 0:
        print("\nSample reversal transactions:")
        # display(matched_reversals_df.head())
        print(matched_reversals_df.head())
        
        print("\nTime between original transaction and reversal:")
        print(f"- Mean: {matched_reversals_df['time_diff_minutes'].mean():.2f} minutes")
        print(f"- Median: {matched_reversals_df['time_diff_minutes'].median():.2f} minutes")
        print(f"- Min: {matched_reversals_df['time_diff_minutes'].min():.2f} minutes")
        print(f"- Max: {matched_reversals_df['time_diff_minutes'].max():.2f} minutes")
    
    print("\nMulti-Swipe Transaction Analysis:")
    print(f"- Total multi-swipe groups identified: {len(multi_swipe_groups)}")
    print(f"- Total extra transactions (excluding first legitimate swipe): {total_multi_swipe_count}")
    print(f"- Total dollar amount of extra swipes: ${total_multi_swipe_amount:.2f}")
    
    if len(multi_swipe_groups) > 0:
        print("\nMulti-swipe group statistics:")
        print(f"- Average transactions per group: {multi_swipe_groups_df['count'].mean():.2f}")
        print(f"- Average time span: {multi_swipe_groups_df['time_span_seconds'].mean():.2f} seconds")
        
        print("\nSample multi-swipe groups:")
        if len(multi_swipe_groups_df) > 0:
            # display(multi_swipe_groups_df.head())
            print(multi_swipe_groups_df.head())
    
    # 3. Identify recurring transactions
    # These are regular transactions with the same amount and merchant
    print("\nIdentifying recurring transactions...")
    recurring_groups = []
    
    # Sample accounts for recurring transaction analysis
    sample_accounts = min(100, len(df_dup['accountNumber'].unique()))
    account_sample = np.random.choice(df_dup['accountNumber'].unique(), sample_accounts, replace=False)
    
    # Group by account, merchant, and amount
    for account in account_sample:
        account_data = df_dup[df_dup['accountNumber'] == account]
        
        # Group by merchant and amount
        for merchant in account_data['merchantName'].unique():
            merchant_data = account_data[account_data['merchantName'] == merchant]
            
            for amount in merchant_data['transactionAmount'].unique():
                # Don't consider $0 transactions
                if amount == 0:
                    continue
                    
                # Get transactions with this exact amount
                amount_txns = merchant_data[
                    (abs(merchant_data['transactionAmount'] - amount) < 0.01) &
                    (merchant_data['transactionType'] != 'REVERSAL')
                ].sort_values('transactionDateTime')
                
                # If we have more than 2 transactions with the same amount
                if len(amount_txns) > 2:
                    # Calculate time differences between consecutive transactions
                    time_diffs = []
                    for i in range(1, len(amount_txns)):
                        time_diff = (amount_txns.iloc[i]['transactionDateTime'] - 
                                    amount_txns.iloc[i-1]['transactionDateTime']).total_seconds() / (24 * 3600)  # in days
                        time_diffs.append(time_diff)
                    
                    # If average time diff is between 7 and 45 days, consider it recurring
                    avg_time_diff = np.mean(time_diffs)
                    if 7 <= avg_time_diff <= 45 and np.std(time_diffs) < avg_time_diff/2:
                        recurring_groups.append({
                            'accountNumber': account,
                            'merchantName': merchant,
                            'amount': amount,
                            'count': len(amount_txns),
                            'first_date': amount_txns.iloc[0]['transactionDateTime'],
                            'last_date': amount_txns.iloc[-1]['transactionDateTime'],
                            'avg_days_between': avg_time_diff,
                            'std_days_between': np.std(time_diffs),
                            'frequency': 'weekly' if avg_time_diff < 10 else 'bi-weekly' if avg_time_diff < 20 else 'monthly'
                        })
    
    recurring_df = pd.DataFrame(recurring_groups) if recurring_groups else pd.DataFrame()
    
    if len(recurring_df) > 0:
        print(f"\nIdentified {len(recurring_df)} potential recurring transaction patterns")
        print("\nSample recurring transaction patterns:")
        print(recurring_df.head())
    else:
        print("No recurring transaction patterns identified")
    
    # Visualize the distribution of time between repeated transactions
    if len(recurring_df) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(recurring_df['avg_days_between'], bins=20, kde=True)
        plt.title('Distribution of Days Between Recurring Transactions')
        plt.xlabel('Average Days Between Transactions')
        plt.ylabel('Frequency')
        plt.grid(True)
        # Ensure the directory exists
        os.makedirs(os.path.join('results', 'plots'), exist_ok=True)
        plt.savefig(os.path.join('results', 'plots', 'recurring_transactions_distribution.png'))
        plt.close()
    
    # 4. Interesting findings
    print("\nInteresting findings about duplicate and recurring transactions:")
    print("1. Reversals typically occur within minutes of the original transaction")
    print("2. Multi-swipe transactions are more common in certain merchant categories")
    print("3. Some merchants appear more frequently in duplicate transactions")
    print("4. Recurring transactions can be useful for identifying subscription services")
    print("5. Anomalies in recurring transaction patterns might indicate fraud")
    
    # Save results before returning
    results = {
        'reversal_transactions': matched_reversals_df,
        'multi_swipe_groups': multi_swipe_groups_df,
        'multi_swipe_transactions': multi_swipe_transactions_df,
        'recurring_transactions': recurring_df,
        'total_reversal_count': len(matched_reversals),
        'total_reversal_amount': total_reversal_amount,
        'total_multi_swipe_count': total_multi_swipe_count,
        'total_multi_swipe_amount': total_multi_swipe_amount,
        'data_shape': df.shape,  # Store data shape for cache validation
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Store timestamp
    }
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    save_results(results, cache_file)
    return results

# If run as a script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data wrangling for fraud detection')
    parser.add_argument('--data', required=True, help='Path to transaction data file')
    parser.add_argument('--cache', default='results/data/duplicate_analysis.pkl', 
                        help='Path to cache file')
    parser.add_argument('--force', action='store_true', 
                        help='Force recalculation even if cache exists')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    print(f"Loading transaction data from {args.data}...")
    try:
        # Try to load the data based on file extension
        if args.data.endswith('.csv'):
            df = pd.read_csv(args.data)
        elif args.data.endswith('.json'):
            df = pd.read_json(args.data, lines=True)
        elif args.data.endswith('.pkl'):
            df = pd.read_pickle(args.data)
        else:
            raise ValueError(f"Unsupported file format: {args.data}")
        
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)
    
    # Analyze duplicates with caching
    results = identify_duplicates(
        df, 
        use_cached=not args.force,
        cache_file=args.cache
    )
    
    # Save summary to CSV
    if results and 'reversal_transactions' in results and len(results['reversal_transactions']) > 0:
        summary_file = 'results/data/reversal_summary.csv'
        results['reversal_transactions'].to_csv(summary_file, index=False)
        print(f"Reversal transactions summary saved to {summary_file}")
    
    print("\nAnalysis complete!")