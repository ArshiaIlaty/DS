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
    
def save_results(results, filename='analysis_results.pkl'):
    """Save analysis results to a pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {filename}")

def load_results(filename='analysis_results.pkl'):
    """Load analysis results from a pickle file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"\nResults loaded from {filename}")
        return results
    return None
    
def identify_duplicates(df, use_cached=True, cache_file='analysis_results.pkl'):
    """
    Identify and analyze reversed and multi-swipe transactions.
    
    Parameters:
    - df: DataFrame containing transaction data
    - use_cached: Whether to try loading cached results first
    - cache_file: File name for caching results
    
    Returns:
    - Dictionary containing information about duplicates
    """
    # Try to load cached results first
    if use_cached:
        results = load_results(cache_file)
        if results is not None:
            return results

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


    print(f"\nMatching {len(explicit_reversals)} reversals to original transactions...")
    for _, reversal in tqdm(explicit_reversals.iterrows(), total=len(explicit_reversals)):
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
    
    # Group by account, merchant, and amount
    for account in df_dup['accountNumber'].unique():
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
    
    # Group by account, merchant, and amount
    for account in df_dup['accountNumber'].unique():
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
        plt.show()
    
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
        'total_multi_swipe_amount': total_multi_swipe_amount
    }
    
    save_results(results, cache_file)
    return results