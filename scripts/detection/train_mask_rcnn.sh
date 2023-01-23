EXP_NAME=mask_rcnn
PYTHONPATH=. python3 tools/train_det.py \
              -c configs/detection/models/mask_rcnn_r50_fpn.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              global.pretrained=null