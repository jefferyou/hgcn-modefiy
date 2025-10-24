import os
import sys
import json
import argparse
import numpy as np
import subprocess
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for Hyperbolic GAIN')
    parser.add_argument('-gpu', default='0', help='GPU ID to use')
    parser.add_argument('-dataset', default='osm_inductive',
                        help='Dataset name (osm_transductive, osm_inductive)')
    parser.add_argument('-prefix', default='italy-osm',
                        help='Dataset prefix (linkoping-osm, sweden-osm, venice-osm)')
    parser.add_argument('-trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('-data_dir', default='graph_data', 
                        help='Directory containing datasets')
    parser.add_argument('-output_dir', default='optimization_results', 
                        help='Directory to save optimization results')
    return parser.parse_args()


def objective(trial, args):
    """Optuna objective function to minimize"""
    
    # Define the hyperparameter search space
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    l2 = trial.suggest_float('l2', 1e-5, 1e-3, log=True)
    units = trial.suggest_categorical('units', [64, 128, 256, 512])
    heads = trial.suggest_categorical('heads', [4, 8, 16])
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    c = trial.suggest_categorical('c', [0, 1])  # Trainable curvature or not
    model_size = trial.suggest_categorical('model_size', ['small', 'big'])
    use_global_walks = trial.suggest_categorical('use_global_walks', [0, 1])
    
    if use_global_walks == 1:
        fusion_type = trial.suggest_categorical('fusion_type', ['simple', 'adaptive'])
    else:
        fusion_type = 'simple'  # Default, won't be used when global walks are off
    
    # Fewer epochs for optimization
    epochs = 3000
    patience = 200
    
    # Create a unique trial directory to store logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}_{timestamp}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Log file for this trial
    log_file = os.path.join(trial_dir, "train.log")
    
    # Command to run with the selected hyperparameters
    cmd = [
        "python", "train_hyperbolic_gain.py",
        "-gpu", args.gpu,
        "-dataset", args.dataset,
        "-prefix", args.prefix,
        "-lr", str(lr),
        "-l2", str(l2),
        "-units", str(units),
        "-heads", str(heads),
        "-dropout", str(dropout),
        "-epochs", str(epochs),
        "-patience", str(patience),
        "-c", str(c),
        "-data_dir", args.data_dir,
        "-model_size", model_size,
        "-use_global_walks", str(use_global_walks),
        "-fusion_type", fusion_type
    ]
    
    # Log the command
    with open(log_file, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")
    
    # Run the command and capture output
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output to both console and log file
        with open(log_file, 'a') as f:
            for line in process.stdout:
                print(line, end='')
                f.write(line)
        
        process.wait()
        
        # Check if the process completed successfully
        if process.returncode != 0:
            print(f"Error in trial {trial.number}. Check log: {log_file}")
            return float('inf')  # Return a large value to indicate failure
        
        # Look for the results file
        results_dir = 'results'
        result_files = [f for f in os.listdir(results_dir) 
                      if f.startswith(f"hyperbolic_gain_{args.dataset}_{args.prefix}_") 
                      and f.endswith(".json")]
        
        # Sort by modification time (newest first)
        result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        
        if not result_files:
            print(f"No result file found for trial {trial.number}")
            return float('inf')
        
        # Read the most recent result file
        with open(os.path.join(results_dir, result_files[0]), 'r') as f:
            result_data = json.load(f)
        
        # Copy the result file to the trial directory
        with open(os.path.join(trial_dir, "result.json"), 'w') as f:
            json.dump(result_data, f, indent=4)
        
        # Extract relevant metrics
        val_loss = result_data.get('val_loss', float('inf'))
        val_acc = result_data.get('val_acc', 0.0)
        test_acc = result_data.get('test_acc', 0.0)
        test_macro_f1 = result_data.get('test_macro_f1', 0.0)
        
        # Save all metrics as trial user attributes
        trial.set_user_attr('val_loss', val_loss)
        trial.set_user_attr('val_acc', val_acc)
        trial.set_user_attr('test_acc', test_acc)
        trial.set_user_attr('test_macro_f1', test_macro_f1)
        
        # Return negative accuracy as we want to maximize accuracy (but Optuna minimizes)
        return -val_acc
        
    except Exception as e:
        print(f"Exception in trial {trial.number}: {str(e)}")
        with open(log_file, 'a') as f:
            f.write(f"\nException: {str(e)}")
        return float('inf')


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up the study
    study_name = f"{args.dataset}_{args.prefix}_optimization"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = os.path.join(args.output_dir, f"{study_name}_{timestamp}.db")
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # We'll minimize negative accuracy
        sampler=TPESampler(seed=42),
        storage=f"sqlite:///{db_path}"
    )
    
    # Run optimization
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)
    
    # Print optimization results
    print("\n" + "="*50)
    print("Hyperparameter Optimization Results")
    print("="*50)
    
    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (negative val_acc): {best_trial.value:.4f}")
    print(f"  Actual val_acc: {-best_trial.value:.4f}")
    print(f"  Test accuracy: {best_trial.user_attrs.get('test_acc', 'N/A'):.4f}")
    print(f"  Test macro F1: {best_trial.user_attrs.get('test_macro_f1', 'N/A'):.4f}")
    
    print("\nBest hyperparameters:")
    for param_name, param_value in best_trial.params.items():
        print(f"  {param_name}: {param_value}")
    
    # Save optimization results
    results_file = os.path.join(args.output_dir, f"optimization_results_{timestamp}.json")
    
    # Gather all trials information
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_data = {
                'trial_number': trial.number,
                'params': trial.params,
                'val_acc': -trial.value if trial.value != float('inf') else 0.0,
                'test_acc': trial.user_attrs.get('test_acc', 0.0),
                'test_macro_f1': trial.user_attrs.get('test_macro_f1', 0.0),
                'val_loss': trial.user_attrs.get('val_loss', float('inf'))
            }
            trials_data.append(trial_data)
    
    # Save all results
    results = {
        'study_name': study_name,
        'dataset': args.dataset,
        'prefix': args.prefix,
        'n_trials': args.trials,
        'completed_trials': len(trials_data),
        'best_trial': {
            'trial_number': best_trial.number,
            'params': best_trial.params,
            'val_acc': -best_trial.value if best_trial.value != float('inf') else 0.0,
            'test_acc': best_trial.user_attrs.get('test_acc', 0.0),
            'test_macro_f1': best_trial.user_attrs.get('test_macro_f1', 0.0),
            'val_loss': best_trial.user_attrs.get('val_loss', float('inf'))
        },
        'all_trials': trials_data
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Generate best configuration script
    best_config_script = os.path.join(args.output_dir, f"best_config_train_script_{timestamp}.sh")
    
    with open(best_config_script, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Best hyperparameter configuration found by optimization\n\n")
        
        cmd = [
            "python", "train_hyperbolic_gain.py",
            "-gpu", args.gpu,
            "-dataset", args.dataset,
            "-prefix", args.prefix,
            "-lr", str(best_trial.params['lr']),
            "-l2", str(best_trial.params['l2']),
            "-units", str(best_trial.params['units']),
            "-heads", str(best_trial.params['heads']),
            "-dropout", str(best_trial.params['dropout']),
            "-c", str(best_trial.params['c']),
            "-data_dir", args.data_dir,
            "-model_size", best_trial.params['model_size'],
            "-use_global_walks", str(best_trial.params['use_global_walks'])
        ]
        
        if best_trial.params['use_global_walks'] == 1:
            cmd.extend(["-fusion_type", best_trial.params['fusion_type']])
        
        f.write(" ".join(cmd))
    
    # Make the script executable
    os.chmod(best_config_script, 0o755)
    
    print(f"\nBest configuration script saved to: {best_config_script}")
    print("\nRun this script to train the model with the best hyperparameters.")


if __name__ == "__main__":
    main()
