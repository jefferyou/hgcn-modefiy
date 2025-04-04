import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    print(f"Command: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Stream the output as it comes
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"Error: {description} failed with return code {process.returncode}")
            return False
        
        print(f"{description} completed successfully.")
        return True
    
    except Exception as e:
        print(f"Error executing command: {str(e)}")
        return False

def verify_data(args):
    """Verify OSM data"""
    command = [
        "python", "verify_osm_data.py",
        "-dataset", args.dataset,
        "-prefix", args.prefix,
        "-data_dir", args.data_dir
    ]
    
    return run_command(command, "Verifying OSM data")


def train_model(args):
    """Train a single HAT model"""
    command = [
        "python", "train_osm_hat.py",
        "-gpu", args.gpu,
        "-dataset", args.dataset,
        "-prefix", args.prefix,
        "-model", args.model,
        "-units", str(args.units),
        "-heads", str(args.heads),
        "-lr", str(args.lr),
        "-dropout", str(args.dropout),
        "-c", str(args.curvature),
        "-data_dir", args.data_dir
    ]

    return run_command(command, "Training a single HAT model")

def run_experiments(args):
    """Run hyperparameter optimization"""
    # For 'all' mode, use a reduced hyperparameter grid
    if args.mode == "all":
        print("Using a reduced hyperparameter grid for the 'all' mode...")
        
        # Create a temporary file with reduced hyperparameter grid
        temp_file = f"run_osm_experiments_temp_{int(time.time())}.py"
        with open("run_osm_experiments.py", "r") as f:
            content = f.read()
        
        # Replace hyperparameter grids with smaller ones
        reduced_content = (
            content.replace("'lr': [0.005, 0.001, 0.0005]", "'lr': [0.005]")
            .replace("'units': [64, 128, 256]", "'units': [64]")
            .replace("'heads': [4, 8]", "'heads': [8]")
            .replace("'dropout': [0.2, 0.5]", "'dropout': [0.2]")
        )
        
        with open(temp_file, "w") as f:
            f.write(reduced_content)
        
        # Run the modified script
        command = [
            "python", temp_file,
            "-gpu", args.gpu,
            "-data_dir", args.data_dir
        ]
        
        result = run_command(command, "Running reduced hyperparameter optimization")
        
        # Remove temporary file
        try:
            os.remove(temp_file)
        except:
            print(f"Warning: Could not remove temporary file {temp_file}")
        
        return result
    else:
        # Run the full hyperparameter optimization
        command = [
            "python", "run_osm_experiments.py",
            "-gpu", args.gpu,
            "-data_dir", args.data_dir
        ]
        
        return run_command(command, "Running hyperparameter optimization")

def analyze_results(args):
    """Analyze experiment results"""
    command = [
        "python", "analyze_results.py",
        "-results_dir", args.results_dir
    ]
    
    return run_command(command, "Analyzing results")

def main():
    parser = argparse.ArgumentParser(description='HAT on OSM Pipeline Runner (Python version)')

    parser.add_argument('-d', '--dataset', default='osm_transductive',
                        help='Dataset type: osm_transductive or osm_inductive (default: osm_transductive)')
    parser.add_argument('-p', '--prefix', default='linkoping-osm',
                        help='Dataset prefix: linkoping-osm or sweden-osm (default: linkoping-osm)')
    parser.add_argument('-g', '--gpu', default='0',
                        help='GPU ID to use (default: 0)')
    parser.add_argument('-m', '--mode', default='all',
                        choices=['all', 'verify', 'train', 'experiment', 'analyze'],
                        help='Pipeline mode: all, verify, train, experiment, or analyze (default: all)')
    parser.add_argument('--data_dir', default='graph_data',
                        help='Directory containing the datasets (default: graph_data)')
    parser.add_argument('--results_dir', default='results',
                        help='Directory for storing results (default: results)')

    # Training parameters for single model
    parser.add_argument('--model', default='hat', choices=['hat', 'gain', 'hybrid'],
                        help='Model type: hat, gain, or hybrid (default: hat)')
    parser.add_argument('--units', type=int, default=64,
                        help='Number of hidden units for single model training (default: 64)')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads for single model training (default: 8)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate for single model training (default: 0.005)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for single model training (default: 0.2)')
    parser.add_argument('--curvature', type=int, default=1,
                        help='Curvature training flag: 0=fixed, 1=trainable (default: 1)')
    
    args = parser.parse_args()
    
    # Check for required directories
    if not os.path.isdir(args.data_dir):
        print(f"Error: '{args.data_dir}' directory not found.")
        print("Please make sure the OSM datasets are properly organized.")
        return 1
    
    # Create output directories if they don't exist
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Print settings
    print("=" * 50)
    print("HAT on OSM Pipeline Runner (Python version)")
    print("=" * 50)
    print("Settings:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Prefix: {args.prefix}")
    print(f"  GPU: {args.gpu}")
    print(f"  Mode: {args.mode}")
    print("=" * 50)
    
    # Execute pipeline based on mode
    success = True
    
    if args.mode == 'all' or args.mode == 'verify':
        print("[1/4] Verifying OSM data...")
        success = verify_data(args)
        if not success and args.mode == 'all':
            print("Data verification failed. Aborting pipeline.")
            return 1
        print("-" * 40)
    
    if (args.mode == 'all' or args.mode == 'train') and success:
        print("[2/4] Training a single HAT model...")
        success = train_model(args)
        if not success and args.mode == 'all':
            print("Model training failed. Aborting pipeline.")
            return 1
        print("-" * 40)
    
    if (args.mode == 'all' or args.mode == 'experiment') and success:
        print("[3/4] Running hyperparameter optimization...")
        print("This may take a long time. You can monitor progress in the logs directory.")
        success = run_experiments(args)
        if not success and args.mode == 'all':
            print("Hyperparameter optimization failed. Aborting pipeline.")
            return 1
        print("-" * 40)
    
    if (args.mode == 'all' or args.mode == 'analyze') and success:
        print("[4/4] Analyzing results...")
        success = analyze_results(args)
        if not success and args.mode == 'all':
            print("Results analysis failed.")
            return 1
        print("-" * 40)
    
    # Final message
    print("=" * 50)
    if success:
        print("Pipeline execution completed successfully!")
    else:
        print("Pipeline execution completed with errors.")
    
    print("=" * 50)
    print("Outputs:")
    print(f"  - Visualizations: ./visualizations/")
    print(f"  - Results: ./{args.results_dir}/")
    print(f"  - Logs: ./logs/")
    print("=" * 50)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
