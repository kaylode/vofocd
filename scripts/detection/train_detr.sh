EXP_NAME=detr
PYTHONPATH=. python3 tools/train_det.py \
              -c configs/detection/detr_r50.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False