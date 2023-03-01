MODEL_NAME=$1
FOLD_ID=$2
EXP_NAME=detr_${MODEL_NAME}_fold${FOLD_ID}_det
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 tools/detection/train_det.py \
              -c configs/detection/models/detr_custom.yaml \
              -o global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              model.args.backbone_name=${MODEL_NAME} \
              global.pretrained=weights/detr-r50-e632da11.pth \
              data.dataset.train.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_train.json \
              data.dataset.val.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_val.json