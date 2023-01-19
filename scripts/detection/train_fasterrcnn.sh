EXP_NAME=detr
PYTHONPATH=. python3 tools/train_det.py \
              -c configs/detection/faster_rcnn_R_50_FPN_1x.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              global.pretrained=null