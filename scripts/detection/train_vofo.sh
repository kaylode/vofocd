FOLD_ID=$1
EXP_NAME=vofo_pico_freezedetr
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 tools/detection/train_det.py \
              -c configs/detection/models/vofo_detr.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              data.dataset.train.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_train.json \
              data.dataset.val.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_val.json
