import os
import subprocess
import itertools
import json
import argparse
from datetime import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Hyperbolic GAIN experiments')

    # GPU settings
    parser.add_argument('--gpu', type=str, default='0',
                        help='Which GPU to use')

    # Experiment settings
    parser.add_argument('--mode', type=str, default='both',
                        choices=['supervised', 'unsupervised', 'both'],
                        help='Which training mode to run')
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['osm_transductive', 'osm_inductive', 'both'],
                        help='Which dataset to use')
    parser.add_argument('--experiments_file', type=str, default=None,
                        help='Path to JSON file with experiment configurations')

    # Output settings
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Directory for experiment outputs')

    return parser.parse_args()


def get_default_experiments():
    """Get default experiment configurations."""
    # Default hyperparameter grid
    hyperparams = {
        'hidden_dims': ['64,64', '128,128', '256,256'],
        'num_heads': ['4,1', '8,1'],
        'learning_rate': [0.005, 0.001, 0.0005],
        'dropout': [0.2, 0.5],
        'curvature_trainable': [True, False],
        'initial_curvature': [0.5, 1.0, 2.0]
    }

    # Generate all combinations
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())

    experiments = []
    for combo in itertools.product(*values):
        experiment = {keys[i]: combo[i] for i in range(len(keys))}
        experiments.append(experiment)

    return experiments


def load_experiments_from_file(file_path):
    """Load experiment configurations from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def run_experiment(gpu, dataset, prefix, mode, experiment, output_dir):
    """Run a single experiment with given parameters."""
    # Create command
    cmd = [
        'python', 'train.py',
        '--gpu', gpu,
        '--dataset', dataset,
        '--prefix', prefix,
        '--mode', mode,
        '--hidden_dims', experiment['hidden_dims'],
        '--num_heads', experiment['num_heads'],
        '--learning_rate', str(experiment['learning_rate']),
        '--dropout', str(experiment['dropout']),
        '--curvature_trainable', str(experiment['curvature_trainable']),
        '--initial_curvature', str(experiment['initial_curvature']),
        '--output_dir', os.path.join(output_dir, f"{dataset}_{prefix}_{mode}")
    ]

    # Convert all arguments to strings
    cmd = [str(c) for c in cmd]

    # Run the command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Return stdout and stderr
    return result.stdout, result.stderr


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Determine datasets and prefixes
    if args.dataset == 'both':
        datasets = ['osm_transductive', 'osm_inductive']
    else:
        datasets = [args.dataset]

    prefixes = {
        'osm_transductive': ['linkoping-osm'],
        'osm_inductive': ['sweden-osm']
    }

    # Determine modes
    if args.mode == 'both':
        modes = ['supervised', 'unsupervised']
    else:
        modes = [args.mode]

    # Load experiments
    if args.experiments_file:
        experiments = load_experiments_from_file(args.experiments_file)
    else:
        experiments = get_default_experiments()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"experiments_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save experiment configurations
    with open(os.path.join(output_dir, 'experiments.json'), 'w') as f:
        json.dump(experiments, f, indent=4)

    # Run experiments
    total_experiments = len(datasets) * sum(len(prefixes[d]) for d in datasets) * len(modes) * len(experiments)
    experiment_count = 0

    print(f"Running {total_experiments} experiments...")

    # Create log file
    log_file = os.path.join(output_dir, 'experiments.log')

    with open(log_file, 'w') as log:
        log.write(f"Starting {total_experiments} experiments at {timestamp}\n\n")

        # Iterate over all combinations
        for dataset in datasets:
            for prefix in prefixes[dataset]:
                for mode in modes:
                    for i, experiment in enumerate(experiments):
                        experiment_count += 1
                        log.write(f"Experiment {experiment_count}/{total_experiments}:\n")
                        log.write(f"  Dataset: {dataset}, Prefix: {prefix}, Mode: {mode}\n")
                        log.write(f"  Parameters: {experiment}\n")

                        # Create experiment directory
                        exp_dir = os.path.join(output_dir, f"{dataset}_{prefix}_{mode}_{i}")
                        os.makedirs(exp_dir, exist_ok=True)

                        # Run experiment
                        try:
                            stdout, stderr = run_experiment(
                                args.gpu, dataset, prefix, mode, experiment, exp_dir
                            )

                            # Save stdout and stderr
                            with open(os.path.join(exp_dir, 'stdout.log'), 'w') as f:
                                f.write(stdout)
                            with open(os.path.join(exp_dir, 'stderr.log'), 'w') as f:
                                f.write(stderr)

                            # Extract and save results
                            log.write("  Completed successfully.\n\n")
                        except Exception as e:
                            log.write(f"  Error: {str(e)}\n\n")

    print(f"All experiments completed. Logs saved to {log_file}")


if __name__ == "__main__":
    try:
        print("Starting Hyperbolic GAIN training...")
        main()
        print("Training completed successfully!")
    except Exception as e:
        import traceback

        print(f"ERROR: {str(e)}")
        traceback.print_exc()