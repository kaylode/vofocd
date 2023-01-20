CONFIG_PATH=$1
WEIGHT_PATH=$2
PYTHONPATH=. python3 tools/infer_det.py \
              -c $CONFIG_PATH \
              -o global.save_dir=runs \
              global.weights=$WEIGHT_PATH