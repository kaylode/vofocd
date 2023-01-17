EXP_NAME=medtex_basic
PYTHONPATH=. python3 tools/train.py \
              -c configs/medtex_basic.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=True \
              global.pretrained_teacher=weights/convnext_small_teacher_pt/checkpoints/best.pth