import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data(project_name, entity_name, run_names=None):  
    """
    Get data from wandb
    Usage: data[(exp_name, seed)] = pd.DataFrame
    """ 
    wandb.login()
    api = wandb.Api()

    runs = api.runs(f"{entity_name}/{project_name}")
    filtered_runs = [run for run in runs if run.name in run_names] if run_names else runs

    data = {}
    for run in filtered_runs:
        exp_name = run.config.get('exp_name', 'unknown')
        seed = run.config.get('seed', 'unknown')

        history = run.history(samples=1000000)
        if history is None:
            continue
        
        history['exp_name'] = exp_name
        history['seed'] = seed

        key = (exp_name, seed)
        if key not in data:
            data[key] = []
        data[key].append(history)

    for key in data:
        data[key] = pd.concat(data[key], ignore_index=True)

    return data

def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std = np.std(data)
    stderr = std / np.sqrt(len(data))
    margin = stderr * 1.96
    return mean, mean - margin, mean + margin

def main():
    # ------------------- Argument -------------------
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--entity_name', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--metric_name', type=str, default='eval/episode_reward')
    parser.add_argument('--confidence', type=float, default=0.95)
    
    args = parser.parse_args()
    params = vars(args)

    project_name = params['project_name']
    entity_name = params['entity_name']
    exp_name = params['exp_name']
    metric_name = params['metric_name']
    confidence = params['confidence']
    seeds = [0, 100, 10000]

    # ------------------- Data -------------------
    data = get_data(project_name, entity_name)
    experiment_data = [data[(exp_name, seed)] for seed in seeds if (exp_name, seed) in data]
    combined_data = pd.concat(experiment_data, ignore_index=True)
    grouped = combined_data.groupby('_step')[metric_name].agg(
        mean='mean',
        std='std',
        count='count'
    ).reset_index()
    grouped['lower'] = grouped['mean'] - 1.96 * grouped['std'] / np.sqrt(grouped['count'])
    grouped['upper'] = grouped['mean'] + 1.96 * grouped['std'] / np.sqrt(grouped['count'])

    # ------------------- Plot -------------------
    plt.figure(figsize=(12, 8))
    plt.plot(grouped['_step'], grouped['mean'], label=f"{exp_name} (Seeds: {seeds})", linewidth=2)
    plt.fill_between(grouped['_step'], grouped['lower'], grouped['upper'], alpha=0.2, label="95% Confidence Interval")
    plt.title(f"Metric: {metric_name} for Experiment: {exp_name}")
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()    


if __name__ == "__main__":
    main()
