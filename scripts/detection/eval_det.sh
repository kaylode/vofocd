CONFIG_PATH=$1
WEIGHT_PATH=$2
EXP_NAME=vocal_eval_det
PYTHONPATH=. python3 tools/detection/eval_det.py \
              -c $CONFIG_PATH \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              global.pretrained=$WEIGHT_PATH \