import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze HAT experiment results on OpenStreetMap datasets')
    parser.add_argument('-results_dir', default='results', help='directory containing the result JSON files')
    return parser.parse_args()

def main(args):
    # Find all result files
    result_files = [f for f in os.listdir(args.results_dir) if f.endswith('.json')]
    
    if not result_files:
        print(f"No result files found in {args.results_dir}")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Load results into a list
    results = []
    for file in result_files:
        with open(os.path.join(args.results_dir, file), 'r') as f:
            result = json.load(f)
            results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Print summary statistics
    print("\n===== Summary Statistics =====")
    print(f"Total experiments: {len(df)}")

    # Group by dataset, prefix, and model
    for (dataset, prefix, model), group in df.groupby(['dataset', 'prefix', 'model']):
        print(f"\n----- {dataset} | {prefix} | {model} -----")
        print(f"Number of experiments: {len(group)}")

        # Best by accuracy
        best_acc = group.loc[group['test_acc'].idxmax()]
        print(f"\nBest by accuracy (acc={best_acc['test_acc']:.4f}):")
        print(f"  lr={best_acc['lr']}, units={best_acc['units']}, heads={best_acc['heads']}, "
              f"dropout={best_acc['dropout']}, c={best_acc['curvature']:.4f}")
        print(f"  macro_f1={best_acc['test_macro_f1']:.4f}, micro_f1={best_acc['test_micro_f1']:.4f}")

        # Best by macro F1
        best_macro = group.loc[group['test_macro_f1'].idxmax()]
        print(f"\nBest by macro F1 (macro_f1={best_macro['test_macro_f1']:.4f}):")
        print(f"  lr={best_macro['lr']}, units={best_macro['units']}, heads={best_macro['heads']}, dropout={best_macro['dropout']}, c={best_macro['curvature']:.4f}")
        print(f"  acc={best_macro['test_acc']:.4f}, micro_f1={best_macro['test_micro_f1']:.4f}")
        
        # Best by micro F1
        best_micro = group.loc[group['test_micro_f1'].idxmax()]
        print(f"\nBest by micro F1 (micro_f1={best_micro['test_micro_f1']:.4f}):")
        print(f"  lr={best_micro['lr']}, units={best_micro['units']}, heads={best_micro['heads']}, dropout={best_micro['dropout']}, c={best_micro['curvature']:.4f}")
        print(f"  acc={best_micro['test_acc']:.4f}, macro_f1={best_micro['test_macro_f1']:.4f}")
    
    # Analyze the effect of curvature
    plt.figure(figsize=(12, 10))
    
    for i, metric in enumerate(['test_acc', 'test_macro_f1', 'test_micro_f1']):
        plt.subplot(3, 1, i+1)
        
        for trainable in [0, 1]:
            trainable_df = df[df['curvature'] == trainable]
            if len(trainable_df) > 0:
                plt.scatter(
                    trainable_df['curvature'], 
                    trainable_df[metric], 
                    label=f"Trainable curvature: {bool(trainable)}",
                    alpha=0.7
                )
        
        plt.xlabel('Curvature value')
        plt.ylabel(metric.replace('_', ' '))
        plt.title(f'Effect of curvature on {metric.replace("_", " ")}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'curvature_effect.png'))
    print(f"\nCurvature effect plot saved to {os.path.join(args.results_dir, 'curvature_effect.png')}")
    
    # Plot comparison of hyperparameters
    params = ['lr', 'units', 'heads', 'dropout']
    metrics = ['test_acc', 'test_macro_f1', 'test_micro_f1']
    
    for param in params:
        plt.figure(figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i+1)
            
            for dataset, marker in zip(['osm_transductive', 'osm_inductive'], ['o', 'x']):
                dataset_df = df[df['dataset'] == dataset]
                if len(dataset_df) > 0:
                    for prefix in dataset_df['prefix'].unique():
                        prefix_df = dataset_df[dataset_df['prefix'] == prefix]
                        
                        # Group by the parameter and calculate mean and std
                        param_stats = prefix_df.groupby(param)[metric].agg(['mean', 'std']).reset_index()
                        
                        plt.errorbar(
                            param_stats[param], 
                            param_stats['mean'], 
                            yerr=param_stats['std'],
                            label=f"{dataset} - {prefix}",
                            marker=marker,
                            capsize=5
                        )
            
            plt.xlabel(param)
            plt.ylabel(metric.replace('_', ' '))
            plt.title(f'Effect of {param} on {metric.replace("_", " ")}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, f'{param}_effect.png'))
        print(f"{param} effect plot saved to {os.path.join(args.results_dir, f'{param}_effect.png')}")
    
    # Save the full results table
    csv_file = os.path.join(args.results_dir, 'all_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"\nFull results saved to {csv_file}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
