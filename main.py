# main.py
import pandas as pd
import json
import argparse
from data_loader import load_data, describe_data
from visualization import plot_transaction_amounts
from data_wrangling import identify_duplicates
from modeling import preprocess_data, build_fraud_model

def main():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Analysis')
    parser.add_argument('--data', required=True, help='Path to the data file')
    parser.add_argument('--no-cache', action='store_true', help='Force recomputation without using cached results')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating plots')
    args = parser.parse_args()

    # Load the data
    print("Loading data...")
    df = load_data(args.data)
    
    # Analyze duplicates with caching
    print("\nAnalyzing duplicates...")
    duplicates = identify_duplicates(df, use_cached=not args.no_cache)
    
    # Describe data
    print("\nGenerating data description...")
    describe_data(df)
    
    # Visualize transaction amounts (optional)
    if not args.skip_plots:
        print("\nGenerating visualizations...")
        plot_transaction_amounts(df)
    
    # Preprocess and build model with caching
    print("\nPreprocessing data...")
    preprocessed_df = preprocess_data(df, use_cached=not args.no_cache)
    
    print("\nBuilding and evaluating models...")
    model_results = build_fraud_model(preprocessed_df, use_cached=not args.no_cache)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()