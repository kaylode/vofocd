MODEL_NAME=$1
FOLD_ID=$2
EXP_NAME=detr_${MODEL_NAME}_fold${FOLD_ID}_det
PYTHONPATH=. python3 tools/detection/train_det.py \
              -c configs/detection/models/detr_custom.yaml \
              -o global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              model.args.backbone_name=${MODEL_NAME} \
              global.pretrained=/mnt/4TBSSD/pmkhoi/isbi/runs/detr_convnext_pico_fold0_det_0/checkpoints/last.pth \
              data.dataset.train.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_train.json \
              data.dataset.val.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_val.json