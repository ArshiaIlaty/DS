"""
Visualization Module for Credit Card Fraud Detection
Functions for plotting transaction data and analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set plotting styles
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def plot_transaction_amounts(df):
    """
    Create visualizations for the transaction amount distribution.
    
    Parameters:
    - df: DataFrame containing transaction data
    """
    print("\n" + "="*80)
    print("TRANSACTION AMOUNT ANALYSIS")
    print("="*80)
    
    # Create a copy to avoid modifying the original DataFrame
    df_plot = df.copy()
    
    # Basic statistics on transaction amounts
    print("\nTransaction Amount Statistics:")
    stats = df_plot['transactionAmount'].describe().to_dict()
    for stat, value in stats.items():
        print(f"- {stat}: {value:.2f}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Histogram
    sns.histplot(df_plot['transactionAmount'], bins=50, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Transaction Amount Distribution')
    axes[0, 0].set_xlabel('Transaction Amount ($)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Log-scale histogram for better visibility of the distribution
    sns.histplot(df_plot['transactionAmount'], bins=50, kde=True, log_scale=True, ax=axes[0, 1])
    axes[0, 1].set_title('Transaction Amount Distribution (Log Scale)')
    axes[0, 1].set_xlabel('Transaction Amount ($)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Box plot
    sns.boxplot(y=df_plot['transactionAmount'], ax=axes[1, 0])
    axes[1, 0].set_title('Transaction Amount Box Plot')
    axes[1, 0].set_ylabel('Transaction Amount ($)')
    
    # Create customized bins for amount categories
    amount_bins = [0, 10, 25, 50, 100, 200, 500, 1000, float('inf')]
    bin_labels = ['$0-10', '$10-25', '$25-50', '$50-100', 
                 '$100-200', '$200-500', '$500-1K', '$1K+']
    
    df_plot['AmountCategory'] = pd.cut(df_plot['transactionAmount'], 
                                      bins=amount_bins, 
                                      labels=bin_labels, 
                                      right=False)
    
    # Count plot of transaction amount categories
    category_counts = df_plot['AmountCategory'].value_counts().sort_index()
    axes[1, 1].bar(category_counts.index, category_counts.values)
    axes[1, 1].set_title('Transaction Amount Categories')
    axes[1, 1].set_xlabel('Amount Category')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(category_counts.values):
        axes[1, 1].text(i, v + 5, f'{v}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis by merchant category
    print("\nTransaction Amount Statistics by Merchant Category:")
    category_stats = df_plot.groupby('merchantCategoryCode')['transactionAmount'].agg(
        ['count', 'mean', 'median', 'min', 'max']).sort_values('count', ascending=False)
    # display(category_stats)
    print(category_stats)
    
    # Plot amounts by merchant category
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='merchantCategoryCode', y='transactionAmount', data=df_plot)
    plt.title('Transaction Amounts by Merchant Category')
    plt.xlabel('Merchant Category')
    plt.ylabel('Transaction Amount ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot frequency of common transaction amounts
    plt.figure(figsize=(14, 6))
    top_amounts = df_plot['transactionAmount'].value_counts().head(15)
    sns.barplot(x=top_amounts.index, y=top_amounts.values)
    plt.title('Most Common Transaction Amounts')
    plt.xlabel('Transaction Amount ($)')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Check for interesting patterns or anomalies
    print("\nObservations about transaction amount structure:")
    print("1. The transaction amount distribution is right-skewed with many small transactions and fewer large ones")
    print("2. There appear to be certain standard/common amounts that occur frequently")
    print("3. Different merchant categories show distinct transaction amount patterns")
    print("4. There may be certain recurring subscription amounts (regular fixed payments)")
    print("5. Some merchant categories like food_delivery show very consistent transaction amounts")
    
    return df_plot

def plot_transaction_time_patterns(df):
    """
    Analyze and visualize transaction patterns over time.
    
    Parameters:
    - df: DataFrame containing transaction data
    """
    # Ensure transaction datetime is in correct format
    df_time = df.copy()
    if 'transactionDateTime' in df_time.columns:
        df_time['transactionDateTime'] = pd.to_datetime(df_time['transactionDateTime'])
    else:
        print("Error: transactionDateTime column not found")
        return
    
    # Extract time components
    df_time['txnHour'] = df_time['transactionDateTime'].dt.hour
    df_time['txnDay'] = df_time['transactionDateTime'].dt.day
    df_time['txnDayOfWeek'] = df_time['transactionDateTime'].dt.dayofweek
    df_time['txnMonth'] = df_time['transactionDateTime'].dt.month
    df_time['txnDayName'] = df_time['transactionDateTime'].dt.day_name()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Transactions by hour of day
    hour_counts = df_time['txnHour'].value_counts().sort_index()
    sns.barplot(x=hour_counts.index, y=hour_counts.values, ax=axes[0, 0])
    axes[0, 0].set_title('Transactions by Hour of Day')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Number of Transactions')
    
    # Transactions by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df_time['txnDayName'].value_counts().reindex(day_order)
    sns.barplot(x=day_counts.index, y=day_counts.values, ax=axes[0, 1])
    axes[0, 1].set_title('Transactions by Day of Week')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Number of Transactions')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Transactions by day of month
    day_counts = df_time['txnDay'].value_counts().sort_index()
    sns.barplot(x=day_counts.index, y=day_counts.values, ax=axes[1, 0])
    axes[1, 0].set_title('Transactions by Day of Month')
    axes[1, 0].set_xlabel('Day of Month')
    axes[1, 0].set_ylabel('Number of Transactions')
    
    # Transactions by month
    month_counts = df_time['txnMonth'].value_counts().sort_index()
    sns.barplot(x=month_counts.index, y=month_counts.values, ax=axes[1, 1])
    axes[1, 1].set_title('Transactions by Month')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Number of Transactions')
    
    plt.tight_layout()
    plt.show()
    
    # Heatmap of transactions by hour and day of week
    pivot_table = pd.pivot_table(
        df_time, 
        values='transactionAmount',
        index='txnDayName', 
        columns='txnHour',
        aggfunc='count'
    )
    
    # Reindex to proper day order
    pivot_table = pivot_table.reindex(day_order)
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_table, cmap='YlGnBu', annot=False)
    plt.title('Transaction Frequency by Hour and Day of Week')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.show()
    
    print("\nObservations about transaction time patterns:")
    print("1. Transaction volume peaks at certain hours of the day")
    print("2. There are day-of-week patterns with higher transaction volumes on weekdays/weekends")
    print("3. Certain days of the month show higher transaction activity")
    print("4. Seasonal patterns may be visible in the monthly data")
    
    return df_time

def plot_fraud_analysis(df):
    """
    Create visualizations for fraud analysis.
    
    Parameters:
    - df: DataFrame containing transaction data with fraud indicators
    """
    # Check if fraud column exists
    if 'isFraud' not in df.columns:
        print("Error: isFraud column not found in the data")
        return
    
    # Make a copy to avoid modifying the original
    df_fraud = df.copy()
    
    # Ensure fraud is boolean
    df_fraud['isFraud'] = df_fraud['isFraud'].astype(bool)
    
    # Create figure for fraud analysis
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Fraud distribution
    fraud_counts = df_fraud['isFraud'].value_counts()
    fraud_pct = fraud_counts / len(df_fraud) * 100
    
    axes[0, 0].bar(['Legitimate', 'Fraudulent'], [fraud_counts[False], fraud_counts[True]])
    axes[0, 0].set_title('Distribution of Legitimate vs Fraudulent Transactions')
    axes[0, 0].set_ylabel('Number of Transactions')
    
    for i, v in enumerate([fraud_counts[False], fraud_counts[True]]):
        axes[0, 0].text(i, v + 5, f'{v} ({fraud_pct[i==1]:.2f}%)', ha='center')
    
    # Transaction amount by fraud
    sns.boxplot(x='isFraud', y='transactionAmount', data=df_fraud, ax=axes[0, 1])
    axes[0, 1].set_title('Transaction Amount by Fraud Status')
    axes[0, 1].set_xlabel('Is Fraud')
    axes[0, 1].set_ylabel('Transaction Amount ($)')
    
    # Fraud by merchant category
    fraud_by_category = df_fraud.groupby('merchantCategoryCode')['isFraud'].mean() * 100
    fraud_by_category = fraud_by_category.sort_values(ascending=False)
    
    sns.barplot(x=fraud_by_category.index, y=fraud_by_category.values, ax=axes[1, 0])
    axes[1, 0].set_title('Fraud Rate by Merchant Category')
    axes[1, 0].set_xlabel('Merchant Category')
    axes[1, 0].set_ylabel('Fraud Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Card present vs fraud
    if 'cardPresent' in df_fraud.columns:
        card_present_fraud = df_fraud.groupby('cardPresent')['isFraud'].mean() * 100
        axes[1, 1].bar(['Card Not Present', 'Card Present'], 
                       [card_present_fraud.get(False, 0), card_present_fraud.get(True, 0)])
        axes[1, 1].set_title('Fraud Rate by Card Present Status')
        axes[1, 1].set_xlabel('Card Present')
        axes[1, 1].set_ylabel('Fraud Rate (%)')
        
        for i, v in enumerate([card_present_fraud.get(False, 0), card_present_fraud.get(True, 0)]):
            axes[1, 1].text(i, v + 0.1, f'{v:.2f}%', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: Fraud by transaction type
    print("\nFraud Rate by Transaction Type:")
    fraud_by_type = df_fraud.groupby('transactionType')['isFraud'].agg(['count', 'sum', 'mean'])
    fraud_by_type['fraud_pct'] = fraud_by_type['mean'] * 100
    fraud_by_type = fraud_by_type.sort_values('fraud_pct', ascending=False)
    # display(fraud_by_type)
    print(fraud_by_type)
    
    # Fraud by hour of day
    if 'transactionDateTime' in df_fraud.columns:
        df_fraud['txnHour'] = pd.to_datetime(df_fraud['transactionDateTime']).dt.hour
        
        plt.figure(figsize=(14, 6))
        fraud_by_hour = df_fraud.groupby('txnHour')['isFraud'].mean() * 100
        sns.lineplot(x=fraud_by_hour.index, y=fraud_by_hour.values)
        plt.title('Fraud Rate by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Fraud Rate (%)')
        plt.xticks(range(0, 24))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    print("\nObservations about fraud patterns:")
    print("1. Fraud rates differ significantly by merchant category")
    print("2. Transaction amount distributions show differences between fraudulent and legitimate transactions")
    print("3. Card-not-present transactions generally have higher fraud rates")
    print("4. Certain hours of the day may show elevated fraud activity")
    print("5. Some transaction types have higher fraud rates than others")
    
    return df_fraud