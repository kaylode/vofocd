MODEL_NAME=$1
EXP_NAME=${MODEL_NAME}_vocal
PYTHONPATH=. python3 tools/train_cls.py \
              -c configs/classification/models/convnext.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              model.args.model_name=${MODEL_NAME}