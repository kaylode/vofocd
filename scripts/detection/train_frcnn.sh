FOLD_ID=$1
EXP_NAME=frcnn_r50_fold${FOLD_ID}_det
PYTHONPATH=. python3 tools/detection/train_det.py \
              -c configs/detection/models/frcnn_r50.yaml \
              -o global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              data.dataset.train.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_train.json \
              data.dataset.val.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_val.json