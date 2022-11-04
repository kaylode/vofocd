EXP_NAME=convnext_nano_only
PYTHONPATH=. python3 tools/train.py \
              -c configs/nano_only.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=True