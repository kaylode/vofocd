CONFIG_PATH=$1
WEIGHT_PATH=$2
FOLD_ID=$3
EXP_NAME=$4
PYTHONPATH=. python3 tools/eval_det.py \
              -c $CONFIG_PATH \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              global.pretrained=$WEIGHT_PATH \
              data.dataset.val.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_val.json