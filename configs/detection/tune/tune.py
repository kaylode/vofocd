import os
import optuna
from theseus.opt import Config
from theseus.base.utilities.optuna_tuner import OptunaWrapper
from source.detection.pipeline import DetPipeline
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    config = Config("configs/detection/tune/detr_tune.yaml")
    seed = config['global'].get('seed', 1702)
    config['global']['exp_name'] = 'detr_efficientnetb0_tune'
    database = os.getenv("POSTGRESQL_OPTUNA")

    tuner = OptunaWrapper(storage=database)
    tuner.tune(
        config=config,
        study_name="detr_efficientnetb0_tune",
        pipeline_class=DetPipeline,
        best_key="f1_score",
        n_trials=30,
        direction="maximize",
        save_dir="runs/optuna",
        sampler=optuna.samplers.RandomSampler(seed=seed),
        # pruner=optuna.pruners.MedianPruner()
    )

    
