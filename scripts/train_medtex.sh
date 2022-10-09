EXP_NAME=$1
PYTHONPATH=. python3 tools/train.py \
              -c configs/classification/pipeline_medtex.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=True \
              global.pretrained_teacher=runs/convnext_large_teacher_pt/checkpoints/best.pth