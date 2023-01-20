CONFIG_PATH=$1
WEIGHT_PATH=$2
EXP_NAME=cls_val
PYTHONPATH=. python3 tools/eval_cls.py \
              -c $CONFIG_PATH \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              global.pretrained=$WEIGHT_PATH