EXP_NAME=detr
PYTHONPATH=. python3 tools/train_segm.py \
              -c configs/segmentation/detrsegm.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=True