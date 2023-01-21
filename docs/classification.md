# Vocal Fold Classification

## **Dataset**
...

## **Execution**

Run train scripts:
```
sh scripts/classification/train_cls.sh {model_name} {fold_id}
```

Example: `sh scripts/classification/train_cls.sh convnext_pico 0`

Run eval script:
```
sh scripts/classification/eval_cls.sh {config_path} {weight} {fold_id}
```
Example: `sh scripts/classification/eval_cls.sh runs/convnext_pico/pipeline.yaml runs/convnext_pico/checkpoints/best.pth 0`


## **Resources**

- Please refer to `https://github.com/kaylode/theseus/tree/v1.1.1/theseus/cv/classification` for the core framework.
