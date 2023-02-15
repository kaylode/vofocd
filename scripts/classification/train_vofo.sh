FOLD_ID=$1
EXP_NAME=vofo_cls
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 tools/train_cls.py \
              -c configs/classification/models/vofo_detr.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=True \
              data.dataset.train.args.json_path=data/aim/annotations/annotation_${FOLD_ID}_train.json \
              data.dataset.val.args.json_path=data/aim/annotations/annotation_${FOLD_ID}_val.json
