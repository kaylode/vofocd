EXP_NAME=detr
PYTHONPATH=. python3 tools/train_detr.py \
              -c configs/detection/detr.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=True