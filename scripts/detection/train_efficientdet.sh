EXP_NAME=$1
FOLD_ID=$2
PYTHONPATH=. python3 tools/train_det.py \
              -c configs/detection/models/efficientdet_r50.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              global.pretrained=null \
              data.dataset.train.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_train.json \
              data.dataset.val.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_val.json