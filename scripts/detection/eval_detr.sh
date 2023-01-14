EXP_NAME=detr_val
PYTHONPATH=. python3 tools/eval_det.py \
              -c weights/detr_r50_pt/pipeline.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              global.pretrained=weights/detr_r50_pt/checkpoints/best.pth