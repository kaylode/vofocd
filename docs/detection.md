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

## YoloV5 detection training
- Clone [this YoloV5 repo](https://github.com/LouisDo2108/yolov5/tree/tuanluc) and see the original documentation in [here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- Download the following [dataset](https://drive.google.com/file/d/1IEFvOGnqchbY78tRjMOXW-DitRdJEySL/view?usp=share_link), which was modified to be compatible with the Yolo format.
- Training script example:
```
python train.py \
--img 480 --batch 32 --epochs 300 \
--data /home/htluc/datasets/aim_folds/fold_0/annotations/aim_fold_0.yaml \
--weights yolov5s.pt \
--name 'yolov5s_fold_0' \
```