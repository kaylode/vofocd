MODEL_NAME=$1
EXP_NAME=detr_${MODEL_NAME}
PYTHONPATH=. python3 tools/train_det.py \
              -c configs/detection/models/detr_custom.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              model.args.backbone_name=${MODEL_NAME}