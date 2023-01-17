EXP_NAME=medtex_ersgan_1
PYTHONPATH=. python3 tools/train.py \
              -c configs/medtex_ersgan_1.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=True \
              global.pretrained_teacher=weights/convnext_small_teacher_pt/checkpoints/best.pth