CONFIG_PATH=$1
WEIGHT_PATH=$2
FOLD_ID=$3
EXP_NAME=vocal_eval_cls
PYTHONPATH=. python3 tools/classification/eval_cls.py \
              -c $CONFIG_PATH \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              global.pretrained=$WEIGHT_PATH \
              data.dataset.val.args.json_path=data/aim/annotations/annotation_${FOLD_ID}_val.json
