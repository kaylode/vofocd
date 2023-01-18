MODEL_NAME=$1
EXP_NAME=detr_${MODEL_NAME}_kvasir
PYTHONPATH=. python3 tools/train_det.py \
              -c configs/detection/detr_${MODEL_NAME}.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False