FOLD_ID=$1
EXP_NAME=effdet_r50_fold${FOLD_ID}_det
PYTHONPATH=. python3 tools/detection/train_det.py \
              -c /mnt/4TBSSD/pmkhoi/isbi/weights/3ff1s2s0/pipeline.yaml \
              -o global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              global.resume=/mnt/4TBSSD/pmkhoi/isbi/weights/3ff1s2s0/checkpoints/last.pth \
              global.pretrained=null \
              data.dataset.train.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_train.json \
              data.dataset.val.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_val.json
        