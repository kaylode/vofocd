EXP_NAME=detr_convnext_kvasir
PYTHONPATH=. python3 tools/train_det.py \
              -c configs/detection/detr_convnext.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False