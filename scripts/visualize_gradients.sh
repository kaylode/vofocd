EXP_NAME=$1
PYTHONPATH=. python3 ./tools/visualize_gradients.py \
              -c configs/classification/test.yaml \
              -o global.save_dir=runs/gradients \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              data.dataset.args.image_dir=data/Train \
              global.cfg_transform=weights/convnext_small_teacher_pt/transform.yaml \
              global.weights=weights/convnext_small_teacher_pt/checkpoints/best.pth