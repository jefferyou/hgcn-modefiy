import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Compare HAT, GAIN, and Hyperbolic GAIN models')
    parser.add_argument('-results_dir', default='results', help='Directory containing result files')
    parser.add_argument('-output_dir', default='visualizations', help='Directory for saving visualizations')
    parser.add_argument('-hat_dir', default=None, help='Directory containing HAT results (optional)')
    parser.add_argument('-gain_dir', default=None, help='Directory containing GAIN results (optional)')
    return parser.parse_args()


def load_results(directory, model_type=None):
    """Load results from JSON files in the given directory"""
    results = []
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} not found. Skipping.")
        return []
        
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            try:
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    # Add model type if provided
                    if model_type:
                        data['model_type'] = model_type
                    elif 'hyperbolic_gain' in filename:
                        data['model_type'] = 'Hyperbolic GAIN'
                    elif 'hat' in filename:
                        data['model_type'] = 'HAT'
                    elif 'gain' in filename:
                        data['model_type'] = 'GAIN'
                    else:
                        # Try to infer from content
                        if 'curvature' in data and data['curvature'] > 0:
                            data['model_type'] = 'Hyperbolic GAIN'
                        else:
                            data['model_type'] = 'Unknown'
                    
                    # Add filename for reference
                    data['filename'] = filename
                    
                    results.append(data)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return results


def create_comparison_table(results_list, output_dir):
    """Create a comparison table of model performance"""
    if not results_list:
        print("No results to compare.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Select relevant columns
    cols = ['model_type', 'dataset', 'prefix', 'test_acc', 'test_macro_f1', 'test_micro_f1']
    
    # Filter for columns that exist
    cols = [col for col in cols if col in df.columns]
    
    if not cols:
        print("No common columns found for comparison.")
        return
    
    # Group by model type, dataset, and prefix
    grouped = df.groupby(['model_type', 'dataset', 'prefix'])[
        ['test_acc', 'test_macro_f1', 'test_micro_f1']
    ].agg(['mean', 'std']).reset_index()
    
    # Format the table for better readability
    table = grouped.copy()
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    table.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    print(f"Comparison table saved to {os.path.join(output_dir, 'model_comparison.csv')}")
    
    return table


def plot_performance_comparison(results_list, output_dir):
    """Create visualizations comparing model performance"""
    if not results_list:
        print("No results to visualize.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the visualization style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('colorblind')
    
    # 1. Bar plot comparing accuracy across models
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(['test_acc', 'test_macro_f1', 'test_micro_f1']):
        if metric not in df.columns:
            continue
            
        plt.subplot(1, 3, i+1)
        
        # Group by model type and dataset
        plot_data = df.groupby(['model_type', 'dataset'])[metric].mean().reset_index()
        
        # Create bar plot
        ax = sns.barplot(x='dataset', y=metric, hue='model_type', data=plot_data)
        
        # Add value labels on top of bars
        for j, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')
        
        plt.title(f"{metric.replace('_', ' ').title()}")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    
    # 2. Plot the effect of curvature (for Hyperbolic GAIN)
    if 'curvature' in df.columns:
        hyperbolic_df = df[df['model_type'] == 'Hyperbolic GAIN']
        
        if not hyperbolic_df.empty:
            plt.figure(figsize=(10, 6))
            
            for metric in ['test_acc', 'test_macro_f1', 'test_micro_f1']:
                if metric not in hyperbolic_df.columns:
                    continue
                
                plt.scatter(hyperbolic_df['curvature'], hyperbolic_df[metric], 
                         label=metric.replace('_', ' ').title(),
                         alpha=0.7, s=50)
            
            plt.title('Effect of Curvature on Model Performance')
            plt.xlabel('Curvature Value')
            plt.ylabel('Performance Metric')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, 'curvature_effect.png'), dpi=300, bbox_inches='tight')
    
    # 3. Plot hyperparameter effects
    for param in ['units', 'heads', 'dropout', 'lr']:
        if param not in df.columns:
            continue
            
        plt.figure(figsize=(15, 5))
        
        for i, metric in enumerate(['test_acc', 'test_macro_f1', 'test_micro_f1']):
            if metric not in df.columns:
                continue
                
            plt.subplot(1, 3, i+1)
            
            for model in df['model_type'].unique():
                model_df = df[df['model_type'] == model]
                
                if model_df.empty:
                    continue
                
                sns.lineplot(x=param, y=metric, data=model_df, label=model, marker='o')
            
            plt.title(f"Effect of {param} on {metric.replace('_', ' ').title()}")
            plt.xlabel(param)
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{param}_effect.png'), dpi=300, bbox_inches='tight')
    
    print(f"Performance comparison visualizations saved to {output_dir}")


def main():
    args = parse_args()
    
    # Load Hyperbolic GAIN results
    hyperbolic_gain_results = load_results(args.results_dir, model_type='Hyperbolic GAIN')
    print(f"Loaded {len(hyperbolic_gain_results)} Hyperbolic GAIN results")
    
    # Load HAT results if specified
    hat_results = []
    if args.hat_dir:
        hat_results = load_results(args.hat_dir, model_type='HAT')
        print(f"Loaded {len(hat_results)} HAT results")
    
    # Load GAIN results if specified
    gain_results = []
    if args.gain_dir:
        gain_results = load_results(args.gain_dir, model_type='GAIN')
        print(f"Loaded {len(gain_results)} GAIN results")
    
    # Combine results
    all_results = hyperbolic_gain_results + hat_results + gain_results
    
    if not all_results:
        print("No results found. Please check your result directories.")
        return
    
    # Create comparison table
    create_comparison_table(all_results, args.output_dir)
    
    # Create comparison visualizations
    plot_performance_comparison(all_results, args.output_dir)


if __name__ == "__main__":
    main()
