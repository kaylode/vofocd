EXP_NAME=convnext_nano_unet
PYTHONPATH=. python3 tools/train.py \
              -c configs/nano_unet.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=True