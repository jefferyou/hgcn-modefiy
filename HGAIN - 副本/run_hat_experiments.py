#!/usr/bin/env python
"""
Script to run HAT experiments on all three OSM datasets
This script automates the testing process and collects all results
"""

import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime
import pandas as pd
import numpy as np


class HATExperimentRunner:
    def __init__(self, gpu='0', data_dir='graph_data', output_dir='hat_results'):
        self.gpu = gpu
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define datasets
        self.datasets = [
            {
                'name': 'Italian Cities',
                'dataset_arg': 'italy',
                'type': 'inductive'
            },
            {
                'name': 'Venice',
                'dataset_arg': 'venice',
                'type': 'transductive'
            },
            {
                'name': 'Munich',
                'dataset_arg': 'munich',
                'type': 'transductive'
            }
        ]
        
        # Default hyperparameters
        self.default_params = {
            'lr': 0.005,
            'l2': 0.0001,
            'units': 8,
            'heads': 8,
            'dropout': 0.6,
            'epochs': 200,
            'patience': 100,
            'c': 1  # trainable curvature
        }
        
        # Random baseline results from the image
        self.baseline_results = {
            'Italian Cities': {'micro_f1': 0.125, 'macro_f1': 0.062, 'weighted_f1': 0.098},
            'Venice': {'micro_f1': 0.167, 'macro_f1': 0.083, 'weighted_f1': 0.139},
            'Munich': {'micro_f1': 0.111, 'macro_f1': 0.056, 'weighted_f1': 0.089}
        }
    
    def run_single_experiment(self, dataset_info, params=None):
        """Run HAT on a single dataset"""
        if params is None:
            params = self.default_params
        
        dataset_name = dataset_info['name']
        dataset_arg = dataset_info['dataset_arg']
        
        print(f"\n{'='*60}")
        print(f"Running HAT on {dataset_name} dataset")
        print(f"{'='*60}")
        
        # Build command
        cmd = [
            'python', 'modified_hat_script.py',
            '-gpu', self.gpu,
            '-dataset', dataset_arg,
            '-data_dir', self.data_dir,
            '-lr', str(params['lr']),
            '-l2', str(params['l2']),
            '-units', str(params['units']),
            '-heads', str(params['heads']),
            '-drop', str(params['dropout']),
            '-epochs', str(params['epochs']),
            '-patience', str(params['patience']),
            '-c', str(params['c'])
        ]
        
        # Log file for this experiment
        log_file = os.path.join(self.output_dir, f'hat_{dataset_arg}_{self.timestamp}.log')
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Log file: {log_file}")
        
        # Run the experiment
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as f:
                # Write command to log file
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Start time: {datetime.now()}\n")
                f.write("="*60 + "\n\n")
                
                # Run the process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Stream output to both console and log file
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                
                process.wait()
                
                # Write completion info
                end_time = time.time()
                duration = end_time - start_time
                f.write(f"\n{'='*60}\n")
                f.write(f"End time: {datetime.now()}\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                
                if process.returncode != 0:
                    print(f"Error: Process exited with code {process.returncode}")
                    return None
                
        except Exception as e:
            print(f"Error running experiment: {e}")
            return None
        
        # Find and load the result file
        result_file = self.find_latest_result_file(dataset_arg)
        if result_file:
            with open(result_file, 'r') as f:
                results = json.load(f)
            print(f"Results loaded from: {result_file}")
            return results
        else:
            print(f"Warning: Could not find result file for {dataset_name}")
            return None
    
    def find_latest_result_file(self, dataset_arg):
        """Find the most recently created result file for a dataset"""
        import glob
        
        pattern = os.path.join(self.output_dir, f'hat_{dataset_arg}_*.json')
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # Sort by modification time, newest first
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files[0]
    
    def run_all_experiments(self):
        """Run experiments on all datasets"""
        print(f"Starting HAT experiments at {datetime.now()}")
        print(f"Output directory: {self.output_dir}")
        print(f"Using GPU: {self.gpu}")
        
        all_results = {}
        
        for dataset_info in self.datasets:
            results = self.run_single_experiment(dataset_info)
            if results:
                all_results[dataset_info['name']] = results
            else:
                print(f"Failed to get results for {dataset_info['name']}")
        
        # Save combined results
        self.save_combined_results(all_results)
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def save_combined_results(self, all_results):
        """Save all results to a single file"""
        combined_file = os.path.join(self.output_dir, f'hat_combined_results_{self.timestamp}.json')
        
        combined_data = {
            'timestamp': self.timestamp,
            'parameters': self.default_params,
            'hat_results': all_results,
            'baseline_results': self.baseline_results
        }
        
        with open(combined_file, 'w') as f:
            json.dump(combined_data, f, indent=4)
        
        print(f"\nCombined results saved to: {combined_file}")
    
    def print_summary(self, all_results):
        """Print a formatted summary of results"""
        print("\n" + "="*80)
        print("HAT EXPERIMENTS SUMMARY")
        print("="*80)
        
        # Create results table
        print(f"\n{'Model':<20} {'Dataset':<20} {'Micro-F1':<12} {'Macro-F1':<12} {'Weighted-F1':<12}")
        print("-"*80)
        
        # Print baseline results
        for dataset_name in ['Italian Cities', 'Venice', 'Munich']:
            if dataset_name in self.baseline_results:
                r = self.baseline_results[dataset_name]
                print(f"{'Random Baseline':<20} {dataset_name:<20} "
                      f"{r['micro_f1']:<12.3f} {r['macro_f1']:<12.3f} {r['weighted_f1']:<12.3f}")
        
        print("-"*80)
        
        # Print HAT results
        for dataset_name in ['Italian Cities', 'Venice', 'Munich']:
            if dataset_name in all_results:
                r = all_results[dataset_name]
                print(f"{'HAT':<20} {dataset_name:<20} "
                      f"{r['micro_f1']:<12.3f} {r['macro_f1']:<12.3f} {r['weighted_f1']:<12.3f}")
        
        print("="*80)
        
        # Calculate and print improvements
        print("\nPerformance Improvements (HAT vs Random Baseline):")
        print("-"*60)
        
        improvements = {}
        for dataset_name in ['Italian Cities', 'Venice', 'Munich']:
            if dataset_name in all_results and dataset_name in self.baseline_results:
                hat = all_results[dataset_name]
                baseline = self.baseline_results[dataset_name]
                
                improvements[dataset_name] = {
                    'micro_f1': ((hat['micro_f1'] - baseline['micro_f1']) / baseline['micro_f1']) * 100,
                    'macro_f1': ((hat['macro_f1'] - baseline['macro_f1']) / baseline['macro_f1']) * 100,
                    'weighted_f1': ((hat['weighted_f1'] - baseline['weighted_f1']) / baseline['weighted_f1']) * 100
                }
                
                print(f"\n{dataset_name}:")
                print(f"  Micro-F1: +{improvements[dataset_name]['micro_f1']:.1f}%")
                print(f"  Macro-F1: +{improvements[dataset_name]['macro_f1']:.1f}%")
                print(f"  Weighted-F1: +{improvements[dataset_name]['weighted_f1']:.1f}%")
        
        # Calculate average improvements
        if improvements:
            avg_micro = np.mean([imp['micro_f1'] for imp in improvements.values()])
            avg_macro = np.mean([imp['macro_f1'] for imp in improvements.values()])
            avg_weighted = np.mean([imp['weighted_f1'] for imp in improvements.values()])
            
            print(f"\nAverage Improvements:")
            print(f"  Micro-F1: +{avg_micro:.1f}%")
            print(f"  Macro-F1: +{avg_macro:.1f}%")
            print(f"  Weighted-F1: +{avg_weighted:.1f}%")
        
        # Generate LaTeX table
        self.print_latex_table(all_results)
    
    def print_latex_table(self, all_results):
        """Print results as LaTeX table"""
        print("\n" + "="*60)
        print("LaTeX Table for Paper:")
        print("="*60)
        
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{HAT Model Performance on OSM Datasets}")
        print("\\begin{tabular}{l|l|ccc}")
        print("\\hline")
        print("\\multirow{2}{*}{Model} & \\multirow{2}{*}{Dataset} & \\multicolumn{3}{c}{F1 Score} \\\\")
        print("\\cline{3-5}")
        print(" & & Micro-F1 & Macro-F1 & Weighted-F1 \\\\")
        print("\\hline")
        
        # Baseline results
        for i, dataset in enumerate(['Italian Cities', 'Venice', 'Munich']):
            if dataset in self.baseline_results:
                r = self.baseline_results[dataset]
                if i == 0:
                    print(f"\\multirow{{3}}{{*}}{{Random Baseline}} & {dataset} & "
                          f"{r['micro_f1']:.3f} & {r['macro_f1']:.3f} & {r['weighted_f1']:.3f} \\\\")
                else:
                    print(f" & {dataset} & "
                          f"{r['micro_f1']:.3f} & {r['macro_f1']:.3f} & {r['weighted_f1']:.3f} \\\\")
        
        print("\\hline")
        
        # HAT results
        for i, dataset in enumerate(['Italian Cities', 'Venice', 'Munich']):
            if dataset in all_results:
                r = all_results[dataset]
                if i == 0:
                    print(f"\\multirow{{3}}{{*}}{{HAT}} & {dataset} & "
                          f"{r['micro_f1']:.3f} & {r['macro_f1']:.3f} & {r['weighted_f1']:.3f} \\\\")
                else:
                    print(f" & {dataset} & "
                          f"{r['micro_f1']:.3f} & {r['macro_f1']:.3f} & {r['weighted_f1']:.3f} \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
    
    def run_hyperparameter_search(self, dataset_name, param_grid):
        """Run hyperparameter search on a specific dataset"""
        # This is a placeholder for hyperparameter search functionality
        # You can extend this to do grid search or random search
        pass


def main():
    parser = argparse.ArgumentParser(description='Run HAT experiments on OSM datasets')
    parser.add_argument('--gpu', default='0', help='GPU ID to use')
    parser.add_argument('--data_dir', default='graph_data', help='Directory containing datasets')
    parser.add_argument('--output_dir', default='hat_results', help='Directory for output files')
    parser.add_argument('--lr', type=float, help='Learning rate (overrides default)')
    parser.add_argument('--units', type=int, help='Hidden units (overrides default)')
    parser.add_argument('--heads', type=int, help='Number of heads (overrides default)')
    parser.add_argument('--dropout', type=float, help='Dropout rate (overrides default)')
    parser.add_argument('--epochs', type=int, help='Maximum epochs (overrides default)')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = HATExperimentRunner(
        gpu=args.gpu,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Override default parameters if specified
    if args.lr is not None:
        runner.default_params['lr'] = args.lr
    if args.units is not None:
        runner.default_params['units'] = args.units
    if args.heads is not None:
        runner.default_params['heads'] = args.heads
    if args.dropout is not None:
        runner.default_params['dropout'] = args.dropout
    if args.epochs is not None:
        runner.default_params['epochs'] = args.epochs
    
    # Run all experiments
    results = runner.run_all_experiments()
    
    print("\nAll experiments completed!")
    
    return results


if __name__ == "__main__":
    main()
