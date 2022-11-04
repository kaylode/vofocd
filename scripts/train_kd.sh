EXP_NAME=distillation
PYTHONPATH=. python3 tools/train.py \
              -c configs/distillation.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=True \
              global.pretrained_teacher=weights/convnext_small_teacher_pt/checkpoints/best.pth