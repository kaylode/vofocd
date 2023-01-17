EXP_NAME=kvasir_resnet50
PYTHONPATH=. python3 tools/train_cls.py \
              -c configs/classification/resnet50.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=True