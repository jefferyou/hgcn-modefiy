import os
import subprocess
import itertools
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Run HAT experiments on OpenStreetMap datasets')
    parser.add_argument('-gpu', nargs='?', default='0', help='the ID for GPU')
    parser.add_argument('-data_dir', default='graph_data', help='directory containing the datasets')
    return parser.parse_args()


def main(args):
    # Define experiment configurations
    configurations = {
        'dataset': ['osm_transductive', 'osm_inductive'],
        'prefix': {
            'osm_transductive': ['linkoping-osm'],
            'osm_inductive': ['sweden-osm']
        },
        'model': ['hat', 'gain', 'hybrid'],  # Add model types
        'lr': [0.005, 0.001, 0.0005],
        'units': [64, 128, 256],
        'heads': [4, 8],
        'dropout': [0.2, 0.5],
        'c': [0, 1]  # 0: untrainable curvature; 1: trainable curvature
    }

    # Create directory for experiment logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/osm_experiments_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Generate experiment configurations
    experiments = []
    for dataset in configurations['dataset']:
        prefixes = configurations['prefix'][dataset]
        for prefix in prefixes:
            for model, lr, units, heads, dropout, c in itertools.product(
                    configurations['model'],
                    configurations['lr'],
                    configurations['units'],
                    configurations['heads'],
                    configurations['dropout'],
                    configurations['c']
            ):
                experiments.append({
                    'dataset': dataset,
                    'prefix': prefix,
                    'model': model,
                    'lr': lr,
                    'units': units,
                    'heads': heads,
                    'dropout': dropout,
                    'c': c
                })

    # Run experiments
    for i, exp in enumerate(experiments):
        print(f"Running experiment {i + 1}/{len(experiments)}: {exp}")

        # Create command
        cmd = [
            "python", "train_osm_hat.py",
            "-gpu", args.gpu,
            "-dataset", exp['dataset'],
            "-prefix", exp['prefix'],
            "-model", exp['model'],
            "-lr", str(exp['lr']),
            "-units", str(exp['units']),
            "-heads", str(exp['heads']),
            "-dropout", str(exp['dropout']),
            "-c", str(exp['c']),
            "-data_dir", args.data_dir
        ]

        # Log file
        log_file = f"{log_dir}/exp_{i + 1}_{exp['dataset']}_{exp['prefix']}_{exp['model']}_lr{exp['lr']}_u{exp['units']}_h{exp['heads']}_d{exp['dropout']}_c{exp['c']}.log"

        # Run command
        with open(log_file, 'w') as f:
            print(f"Logging to: {log_file}")
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
    
    print(f"All {len(experiments)} experiments completed!")
    print(f"Logs saved to: {log_dir}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
