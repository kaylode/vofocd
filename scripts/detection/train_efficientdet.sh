EXP_NAME=efficientdet
PYTHONPATH=. python3 tools/train_det.py \
              -c configs/detection/models/efficientdet_d0.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              global.pretrained=null