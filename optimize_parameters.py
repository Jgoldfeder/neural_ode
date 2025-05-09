import os
import sys
import yaml
import argparse
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from pathlib import Path
from typing import List, Dict, Any
import datetime
import numpy as np
from ctf4science.data_module import load_dataset, get_prediction_timesteps, parse_pair_ids, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from naive_baselines import NaiveBaseline

file_dir = Path(__file__).parent

# Update python PATH so that we can load run.py from CTF_NaiveBaselines directly
sys.path.insert(0, str(file_dir))

from run_opt import main as run_opt_main

def sum_results(results):
    """
    Sums all metric values from a results dictionary containing evaluation metrics.
    
    Iterates through all pairs in the results dictionary and sums all metric values
    found in each pair's 'metrics' dictionary. This is used to aggregate
    evaluation metrics from a batch_results.yaml file.

    Args:
        results (dict): A dictionary containing evaluation results.
    
    Returns:
        float: The sum of all metric values across all pairs in the results dictionary.
    """
    total = 0
    for pair_dict in results['pairs']:
        metric_dict = pair_dict['metrics']
        for metric in metric_dict.keys():
            total += metric_dict[metric]
    return total

def create_search_space(tuning_config):
    """
    Create a Ray Tune search space dictionary from the tuning config flie.

    Args:
        tuning_config (dict):
            Dictionary containing the parameter specification with the following keys:
            - 'type': str, either 'float' or 'int' indicating the parameter type
            - 'lower_bound': float/int, the minimum value for the parameter
            - 'upper_bound': float/int, the maximum value for the parameter
            - 'log': bool, whether to sample in log space

    Returns:
        dict: Ray Tune expected search_space dictionary

    Raises:
        Exception:
            If any of the required keys ('type', 'lower_bound', 'upper_bound', 'log')
            are missing from tuning_config for a parameter.
            If the parameter type is neither 'float' nor 'int'.
    """
    search_space = {}
    for name in tuning_config.keys():
        param_dict = tuning_config[name]
        if 'type' not in param_dict:
            raise Exception(f"\'type\' not in {param_dict} keys")

        if param_dict['type'] == "uniform":
            search_space[name] = tune.uniform(param_dict['lower_bound'], param_dict['upper_bound'])
        elif param_dict['type'] == "quniform":
            search_space[name] = tune.quniform(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
        elif param_dict['type'] == "loguniform":
            search_space[name] = tune.loguniform(param_dict['lower_bound'], param_dict['upper_bound'])
        elif param_dict['type'] == "qloguniform":
            search_space[name] = tune.qloguniform(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
        elif param_dict['type'] == "randn":
            search_space[name] = tune.randn(param_dict['lower_bound'], param_dict['upper_bound'])
        elif param_dict['type'] == "qrandn":
            search_space[name] = tune.qrandn(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
        elif param_dict['type'] == "randint":
            search_space[name] = tune.randint(param_dict['lower_bound'], param_dict['upper_bound'])
        elif param_dict['type'] == "qrandint":
            search_space[name] = tune.qrandint(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
        elif param_dict['type'] == "lograndint":
            search_space[name] = tune.lograndint(param_dict['lower_bound'], param_dict['upper_bound'])
        elif param_dict['type'] == "qlograndint":
            search_space[name] = tune.qlograndint(param_dict['lower_bound'], param_dict['upper_bound'], param_dict['q'])
        elif param_dict['type'] == "choice":
            search_space[name] = tune.choice(param_dict['choices'])
        elif param_dict['type'] == "grid":
            search_space[name] = tune.grid(param_dict['grid'])
        else:
            raise Exception(f"Parameter type {param_dict['type']} not supported.")

    return search_space

def generate_config(config, template, name):
    """
    Generates a configuration file with suggested hyperparameter values.

    This function suggests a value for the constant parameter using Raytun's config,
    updates the configuration template with this value, and saves the resulting
    configuration to a YAML file.

    Args:
        config (dict): Dictionary containing selected hyperparameters.
        template (dict): Configuration template dictionary that will be populated with
            the suggested values.
        name (str): Name to use for the output configuration file (without extension).

    Returns:
        dict: updated config file

    Side Effects:
        - Writes a new YAML configuration file to ./config/{name}.yaml
        - Modifies the input template dictionary by adding the suggested constant value
    """
    # Fill out dictionary with hyperparameter values
    for blank_key in config.keys():
        template['model'][blank_key] = config[blank_key]
    # Save config
    config_path = file_dir / 'config' / f'{name}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(template, f)
    return config_path

def main(config_path: str, save_config: bool = True) -> None:
    """
    Main function to generate configuration files and run the naive baseline
    model on specified sub-datasets for hyperparameter optimization.

    Loads configuration, generates specific config files, runs model on training
    and validation set, and performs hyperparameter optimization.

    Args:
        config_path (str): Path to the configuration file.
        save_config (str): Save the final configuration file. (only False in unit tests)
    """
    # Load hyperparameter configuration
    with open(config_path, 'r') as f:
        hp_config = yaml.safe_load(f)
    
    # Separate configuration file from hyperparameters
    hyperparameters = hp_config.pop('hyperparameters')
    blank_config = hp_config.copy()

    # Generate parameter dictionary for Ray Tune
    param_dict = create_search_space(hyperparameters)

    # Define objective for Ray Tune
    def objective(config):
        # Get batch_id
        batch_id = str(tune.get_context().get_trial_id())
        # Fill out dictionary with required values
        batch_id = str(tune.get_context().get_trial_id())
        blank_config['model']['batch_id'] = batch_id
        # Create config file
        config_path = generate_config(config, blank_config, f'hp_config_{batch_id}')
        # Run model
        run_opt_main(config_path)
        # Extract results and clean up files
        config_file = file_dir / 'config' / f'hp_config_{batch_id}.yaml'
        results_file = file_dir / f'results_{batch_id}.yaml'
        with open(results_file, 'r') as f:
            results = yaml.safe_load(f)
        results_file.unlink(missing_ok=True)
        config_file.unlink(missing_ok=True)
        score = sum_results(results)
        # Return score
        return {"score": score}

    # Create Ray Tune object
    tuner = tune.Tuner(objective,
                        param_space=param_dict,
                        tune_config=tune.TuneConfig(
                            #search_alg=OptunaSearch(), # Throws errors (seg faults) but still runs
                            max_concurrent_trials=1,
                            num_samples=blank_config['model']['n_trials'],
                            metric="score",
                            mode="max",
                        ))
    
    # Run optimization
    results = tuner.fit()

    # Obtain best hyperparameter value
    result = results.get_best_result(metric="score", mode="max")
    best_config = result.config
    best_score = result.metrics['score']
    print(f"Best score: {best_score} (params: {best_config})")

    # Save final configuration yaml from hyperparameter optimization
    if not save_config: # Only False when unit testing
        print("Not saving final config file.")
    else:
        pair_ids = ''.join(map(str,blank_config["dataset"]["pair_id"]))
        blank_config['model'].pop('batch_id', None)
        blank_config['model'].pop('n_trials', None)
        blank_config['model'].pop('train_split', None)
        config_path = generate_config(best_config, blank_config, f'config_{blank_config["dataset"]["name"]}_{pair_ids}_optimized')
        print("Final config file saved to:", config_path)
        with open(config_path, 'w') as f:
            yaml.dump(blank_config, f)

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the hyperparameter configuration file.")
    parser.add_argument('save_config', action='store_true', help="Save the final hyperparameter configuration file. Only used when unit testing.")
    args = parser.parse_args()
    main(args.config, args.save_config)