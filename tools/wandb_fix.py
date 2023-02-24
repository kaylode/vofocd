import wandb
import numpy as np

# from tqdm import tqdm

ENTITY = 'hcmus-dcu'
PROJECT = 'aim'
METRIC_NAME = 'Validation/bl_acc'

api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}")

for run in runs:

    if METRIC_NAME not in run.history().columns:
        continue
    values = run.history()[METRIC_NAME].values
    values = values[~np.isnan(values)]

    run.summary[f"{METRIC_NAME}_max"] = np.max(values)

    run.summary.update()