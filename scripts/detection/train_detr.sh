MODEL_NAME=$1
FOLD_ID=$2
EXP_NAME=detr_${MODEL_NAME}
PYTHONPATH=. python3 tools/train_det.py \
              -c configs/detection/models/detr_custom.yaml \
              -o global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              model.args.backbone_name=${MODEL_NAME} \
              data.dataset.train.args.label_path=data/aim/annotations/aim_${FOLD_ID}_train.json \
              data.dataset.val.args.label_path=data/aim/annotations/aim_${FOLD_ID}_val.json