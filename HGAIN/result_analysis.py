import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter optimization results')
    parser.add_argument('-results_dir', required=True,
                        help='Directory containing optimization result files')
    parser.add_argument('-output_dir', default='result_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('-plot_format', default='png', choices=['png', 'pdf', 'svg'],
                        help='Format for saving plots')
    parser.add_argument('-dataset', default=None,
                        help='Filter analysis to specific dataset')
    parser.add_argument('-prefix', default=None,
                        help='Filter analysis to specific prefix')
    return parser.parse_args()


def find_result_files(directory):
    """Find all result files in the given directory and its subdirectories"""
    result_files = []

    # Look for JSON files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json') and not file.startswith('search_configurations'):
                if ('results' in file or 'optimization' in file or
                        file.startswith('grid_search') or
                        file.startswith('hyperbolic_gain')):
                    result_files.append(os.path.join(root, file))

    return result_files


def extract_results(file_path):
    """Extract results from a file based on its format"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Determine the file type and extract results accordingly
    results = []

    # Case 1: Single result file (from train_hyperbolic_gain.py)
    if isinstance(data, dict) and 'val_acc' in data and 'test_acc' in data:
        # Add source file info
        data['source_file'] = os.path.basename(file_path)
        data['source_type'] = 'train_direct'
        results.append(data)

    # Case 2: Optuna results (from hyperparameter_optimization.py)
    elif isinstance(data, dict) and 'best_trial' in data and 'all_trials' in data:
        # Add source file info to all trials
        for trial in data['all_trials']:
            trial['source_file'] = os.path.basename(file_path)
            trial['source_type'] = 'optuna'
            results.append(trial)

    # Case 3: Grid search or parallel search results (list of configs)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and ('val_acc' in item or 'test_acc' in item):
                item['source_file'] = os.path.basename(file_path)
                if 'grid_search' in file_path:
                    item['source_type'] = 'grid_search'
                else:
                    item['source_type'] = 'parallel_search'
                results.append(item)

    return results


def consolidate_results(result_files, filter_dataset=None, filter_prefix=None):
    """Consolidate results from multiple files into a single dataframe"""
    all_results = []

    for file_path in result_files:
        results = extract_results(file_path)
        all_results.extend(results)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Apply filters if specified
    if filter_dataset and 'dataset' in df.columns:
        df = df[df['dataset'] == filter_dataset]

    if filter_prefix and 'prefix' in df.columns:
        df = df[df['prefix'] == filter_prefix]

    return df


def analyze_parameter_importance(df, output_dir, plot_format):
    """Analyze the importance of different hyperparameters"""
    # Create correlation heatmap for numerical parameters
    numeric_params = ['lr', 'l2', 'units', 'heads', 'dropout',
                      'val_acc', 'test_acc', 'test_macro_f1']
    numeric_df = df[numeric_params].copy()

    # Convert units and heads to numeric if not already
    for col in ['units', 'heads']:
        if col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col])

    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Parameter Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'parameter_correlation.{plot_format}'))
    plt.close()

    # Create box plots for categorical parameters
    categorical_params = ['c', 'model_size', 'use_global_walks', 'fusion_type']

    for param in categorical_params:
        if param in df.columns and len(df[param].unique()) > 1:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=param, y='val_acc', data=df)
            plt.title(f'Effect of {param} on Validation Accuracy')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param}_val_acc.{plot_format}'))
            plt.close()

            plt.figure(figsize=(10, 6))
            sns.boxplot(x=param, y='test_acc', data=df)
            plt.title(f'Effect of {param} on Test Accuracy')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param}_test_acc.{plot_format}'))
            plt.close()

    # Create scatter plots for numerical parameters
    for param in ['lr', 'l2', 'units', 'heads', 'dropout']:
        if param in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=param, y='val_acc', data=df, alpha=0.7,
                            hue='source_type' if 'source_type' in df.columns else None)
            plt.title(f'Effect of {param} on Validation Accuracy')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param}_val_acc_scatter.{plot_format}'))
            plt.close()

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=param, y='test_acc', data=df, alpha=0.7,
                            hue='source_type' if 'source_type' in df.columns else None)
            plt.title(f'Effect of {param} on Test Accuracy')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{param}_test_acc_scatter.{plot_format}'))
            plt.close()


def create_summary_report(df, output_dir):
    """Create a summary report of the best configurations"""
    # Find best configurations
    if 'val_acc' in df.columns and len(df) > 0:
        best_val_acc = df.loc[df['val_acc'].idxmax()]
        best_val_acc_index = df['val_acc'].idxmax()
    else:
        best_val_acc = pd.Series({'val_acc': 'N/A'})
        best_val_acc_index = None

    if 'test_acc' in df.columns and len(df) > 0:
        best_test_acc = df.loc[df['test_acc'].idxmax()]
        best_test_acc_index = df['test_acc'].idxmax()
    else:
        best_test_acc = pd.Series({'test_acc': 'N/A'})
        best_test_acc_index = None

    if 'test_macro_f1' in df.columns and len(df) > 0:
        best_test_f1 = df.loc[df['test_macro_f1'].idxmax()]
        best_test_f1_index = df['test_macro_f1'].idxmax()
    else:
        best_test_f1 = pd.Series({'test_macro_f1': 'N/A'})
        best_test_f1_index = None

    # Create summary DataFrame
    summary = pd.DataFrame({
        'Best Val Acc': best_val_acc,
        'Best Test Acc': best_test_acc,
        'Best Test F1': best_test_f1
    }).transpose()

    # Filter columns of interest
    columns_of_interest = [
        'val_acc', 'test_acc', 'test_macro_f1', 'lr', 'l2', 'units', 'heads',
        'dropout', 'c', 'model_size', 'use_global_walks', 'fusion_type',
        'source_type', 'source_file'
    ]

    summary_cols = [col for col in columns_of_interest if col in summary.columns]
    summary = summary[summary_cols]

    # Save summary to file
    summary.to_csv(os.path.join(output_dir, 'best_configurations.csv'))

    # Generate scripts with best configurations
    script_file = os.path.join(output_dir, 'best_configurations.sh')
    with open(script_file, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Generated by analyze_results.py\n")
        f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Best validation accuracy configuration
        if best_val_acc_index is not None:
            row = df.loc[best_val_acc_index]
            f.write("# Best validation accuracy configuration\n")
            command = build_command_from_row(row)
            f.write(command + "\n\n")

        # Best test accuracy configuration
        if best_test_acc_index is not None:
            row = df.loc[best_test_acc_index]
            f.write("# Best test accuracy configuration\n")
            command = build_command_from_row(row)
            f.write(command + "\n\n")

        # Best test F1 configuration
        if best_test_f1_index is not None:
            row = df.loc[best_test_f1_index]
            f.write("# Best test macro F1 configuration\n")
            command = build_command_from_row(row)
            f.write(command + "\n")

    # Make the script executable
    os.chmod(script_file, 0o755)

    return summary


def build_command_from_row(row):
    """Build a command line from a DataFrame row"""
    command = ["python", "train_hyperbolic_gain.py"]

    # Required parameters
    if 'dataset' in row and not pd.isna(row['dataset']):
        command.extend(["-dataset", str(row['dataset'])])

    if 'prefix' in row and not pd.isna(row['prefix']):
        command.extend(["-prefix", str(row['prefix'])])

    # Optional parameters
    param_mapping = {
        'lr': '-lr',
        'l2': '-l2',
        'units': '-units',
        'heads': '-heads',
        'dropout': '-dropout',
        'c': '-c',
        'model_size': '-model_size',
        'use_global_walks': '-use_global_walks',
        'fusion_type': '-fusion_type'
    }

    for param, flag in param_mapping.items():
        if param in row and not pd.isna(row[param]):
            command.extend([flag, str(row[param])])

    return " ".join(command)


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all result files
    result_files = find_result_files(args.results_dir)
    print(f"Found {len(result_files)} result files in {args.results_dir}")

    # Consolidate results
    df = consolidate_results(result_files, args.dataset, args.prefix)
    print(f"Consolidated {len(df)} result entries")

    if len(df) == 0:
        print("No results found. Check your filters or result directory.")
        return

    # Save consolidated results
    df.to_csv(os.path.join(args.output_dir, 'consolidated_results.csv'), index=False)

    # Analyze parameter importance
    analyze_parameter_importance(df, args.output_dir, args.plot_format)

    # Create summary report
    summary = create_summary_report(df, args.output_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("Analysis Summary")
    print("=" * 50)
    print(f"\nTotal configurations analyzed: {len(df)}")

    if 'source_type' in df.columns:
        source_counts = df['source_type'].value_counts()
        print("\nSource types:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")

    print("\nBest configurations:")
    print(summary.to_string())

    print("\nResults saved to:", args.output_dir)
    print("  - Complete results:", os.path.join(args.output_dir, 'consolidated_results.csv'))
    print("  - Best configurations:", os.path.join(args.output_dir, 'best_configurations.csv'))
    print("  - Best configuration script:", os.path.join(args.output_dir, 'best_configurations.sh'))
    print("  - Parameter analysis plots:", args.output_dir)


if __name__ == "__main__":
    main()
