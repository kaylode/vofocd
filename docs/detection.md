# Vocal Fold Detection

## **Dataset**
...

## **Execution**

- To train DETR with convnext backbone, modify the configuration files in `isbi/configs/detection/models/*.yaml`. Then run the following script:
```
sh scripts/detection/train_detr.sh {model_name} {fold_id}
```
Example: `sh scripts/detection/train_detr.sh convnext_pico 0`

Run eval script:
```
sh scripts/detection/eval_detr.sh {config_path} {weight} {fold_id}
```
Example: `sh scripts/detection/eval_detr.sh runs/convnext_pico/pipeline.yaml runs/convnext_pico/checkpoints/best.pth 0`

## Hyperparameter Tuning
- To tune the hyperparameters, modify the configuration in `isbi/configs/detection/tune/detr_tune.yaml`. Then run the following script:
```
PYTHONPATH=. python configs/detection/tune/tune.py
```

## **Resources**

- Please refer to `https://github.com/kaylode/theseus/tree/v1.1.1/theseus/cv/detection` for the core framework.