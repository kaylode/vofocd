CONFIG_PATH=$1
WEIGHT_PATH=$2
EXP_NAME=vocal_infer_det
PYTHONPATH=. python3 tools/detection/infer_det.py \
              -c $CONFIG_PATH \
              -o global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              global.weights=$WEIGHT_PATH