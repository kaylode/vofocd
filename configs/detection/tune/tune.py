import os
from theseus.opt import Config
from theseus.base.utilities.optuna_tuner import OptunaWrapper
from source.detection.pipeline import DetPipeline
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    config = Config("configs/detection/tune/detr_tune.yaml")
    database = os.getenv("POSTGRESQL_OPTUNA")
    tuner = OptunaWrapper(storage=database)

    tuner.tune(
        config=config,
        study_name="detr_convnext_tune",
        pipeline_class=DetPipeline,
        best_key="f1_score",
        n_trials=5,
        direction="maximize",
        save_dir="runs/optuna/",
    )

    
