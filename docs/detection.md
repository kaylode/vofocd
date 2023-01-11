# Vocal Fold Detection

## **Dataset**
...

## **Execution**

- To train DETR with convnext backbone, modify the configuration in `isbi/configs/detection/detr.yaml`. Then run the following script:
```
sh scripts/detection/train_detr.sh
```

## **Resources**

- Please refer to `https://github.com/kaylode/theseus/tree/v1.1.1/theseus/cv/detection` for the core framework.