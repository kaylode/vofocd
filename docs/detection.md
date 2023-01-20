# Vocal Fold Detection

## **Dataset**
...

## **Execution**

- To train DETR with convnext backbone, modify the configuration files in `isbi/configs/detection/models.yaml`. Then run the following script:
```
sh scripts/detection/train_detr.sh
```

## Hyperparameter Tuning
- To tune the hyperparameters, modify the configuration in `isbi/configs/detection/tune/detr_tune.yaml`. Then run the following script:
```
PYTHONPATH=. python configs/detection/tune/tune.py
```

## **Resources**

- Please refer to `https://github.com/kaylode/theseus/tree/v1.1.1/theseus/cv/detection` for the core framework.