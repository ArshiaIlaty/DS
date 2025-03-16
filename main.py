# main.py
import pandas as pd
import json
import argparse
import os
import numpy as np
import time
from pathlib import Path
from data_loader import load_data, describe_data
from visualization import plot_transaction_amounts, plot_transaction_time_patterns, plot_fraud_analysis
from data_wrangling import identify_duplicates
from modeling import run_fraud_detection_pipeline
from utils import create_standard_directories, clean_data_for_analysis

def main():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Analysis')
    parser.add_argument('--data', required=True, help='Path to the data file')
    parser.add_argument('--no-cache', action='store_true', help='Force recomputation without using cached results')
    parser.add_argument('--force-recalculate', action='store_true', help='Force recalculation even if cache exists')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--skip-duplicates', action='store_true', help='Skip duplicate transaction analysis')
    parser.add_argument('--skip-modeling', action='store_true', help='Skip model training and evaluation')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of transactions to sample')
    args = parser.parse_args()

    print(f"Running fraud detection analysis with the following settings:")
    print(f"- Data file: {args.data}")
    print(f"- Sample size: {args.sample_size}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Use cache: {not args.no_cache}")
    print(f"- Force recalculation: {args.force_recalculate}")
    print(f"- Skip plots: {args.skip_plots}")
    print(f"- Skip duplicates analysis: {args.skip_duplicates}")
    print(f"- Skip modeling: {args.skip_modeling}")

    # Create directory structure
    create_standard_directories(args.output_dir)

    # Start timing
    start_time = time.time()

    # Load the data
    print("\nLoading data...")
    df = load_data(args.data, sample_size=args.sample_size)
    
    if df is None:
        print("Error loading data. Exiting...")
        return
    
    # Clean data for analysis
    print("\nCleaning data for analysis...")
    df_clean = clean_data_for_analysis(df)
    
    # Save initial data description
    print("\nGenerating data description...")
    with open(os.path.join(args.output_dir, 'metrics', 'data_description.txt'), 'w') as f:
        f.write("DATA DESCRIPTION\n")
        f.write("="*80 + "\n\n")
        describe_data(df_clean)
    
    # Analyze duplicates with caching (unless skipped)
    if not args.skip_duplicates:
        print("\n" + "="*80)
        print("DUPLICATE TRANSACTION ANALYSIS")
        print("="*80)
        
        cache_file = os.path.join(args.output_dir, 'data', 'duplicate_analysis.pkl')
        duplicates = identify_duplicates(
            df_clean, 
            use_cached=not args.no_cache, 
            cache_file=cache_file,
            force_recalculate=args.force_recalculate
        )
    else:
        print("\nSkipping duplicate transaction analysis as requested.")
    
    # Generate visualizations
    if not args.skip_plots:
        print("\nGenerating visualizations...")
        plot_transaction_amounts(df_clean)
        plot_transaction_time_patterns(df_clean)
        if 'isFraud' in df_clean.columns:
            plot_fraud_analysis(df_clean)
    
    # Run the fraud detection pipeline (unless skipped)
    results = None
    if not args.skip_modeling:
        print("\n" + "="*80)
        print("FRAUD DETECTION PIPELINE")
        print("="*80)
        
        # Include sample size in the cache file name
        sample_size_str = f"_sample_{args.sample_size}" if args.sample_size else ""
        pipeline_cache_file = os.path.join(args.output_dir, f'pipeline_results{sample_size_str}.pkl')
        
        if not args.no_cache and not args.force_recalculate and os.path.exists(pipeline_cache_file):
            try:
                with open(pipeline_cache_file, 'rb') as f:
                    import pickle
                    results = pickle.load(f)
                print(f"\nLoaded cached pipeline results from {pipeline_cache_file}")
                print(f"Using cached model results for sample size {args.sample_size} to avoid recalculation.")
            except Exception as e:
                print(f"Error loading cached pipeline results: {str(e)}. Running pipeline...")
                results = run_fraud_detection_pipeline(
                    data_path=args.data,
                    sample_size=args.sample_size,
                    output_dir=args.output_dir,
                    use_cache=not args.no_cache,
                    force_recalculate=args.force_recalculate
                )
        else:
            # Run the pipeline if no cache or forced recalculation
            results = run_fraud_detection_pipeline(
                data_path=args.data,
                sample_size=args.sample_size,
                output_dir=args.output_dir,
                use_cache=not args.no_cache,
                force_recalculate=args.force_recalculate
            )
            
            # Save pipeline results for future use
            try:
                with open(pipeline_cache_file, 'wb') as f:
                    import pickle
                    pickle.dump(results, f)
                print(f"Pipeline results saved to {pipeline_cache_file}")
            except Exception as e:
                print(f"Error saving pipeline results: {str(e)}")
    else:
        print("\nSkipping model training and evaluation as requested.")
    
    # Save summary of results
    print("\nSaving results summary...")
    with open(os.path.join(args.output_dir, 'metrics', 'analysis_summary.txt'), 'w') as f:
        f.write("FRAUD DETECTION ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data file: {args.data}\n")
        f.write(f"Number of transactions: {len(df_clean)}\n")
        if 'isFraud' in df_clean.columns:
            fraud_rate = (df_clean['isFraud'].sum() / len(df_clean)) * 100
            f.write(f"Fraud rate: {fraud_rate:.2f}%\n\n")
        
        f.write("Results are organized in the following directories:\n")
        f.write("- models/: Trained models and model results\n")
        f.write("- plots/: Visualizations and analysis plots\n")
        f.write("- data/: Preprocessed data and intermediate results\n")
        f.write("- metrics/: Performance metrics and analysis results\n")
    
    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    # Print summary of model results if available
    if results:
        print("\nModel Results Summary:")
        for model_name, model_results in results.items():
            if 'evaluation' in model_results:
                print(f"- {model_name}: F1 Score = {model_results['evaluation'].get('f1', 'N/A'):.4f}")
    
    print("\nAnalysis complete!")
    print(f"Results have been saved to the '{args.output_dir}' directory")
    print("You can find:")
    print(f"- Model files in: {args.output_dir}/models/")
    print(f"- Plots in: {args.output_dir}/plots/")
    print(f"- Processed data in: {args.output_dir}/data/")
    print(f"- Metrics and summaries in: {args.output_dir}/metrics/")

if __name__ == "__main__":
    main()