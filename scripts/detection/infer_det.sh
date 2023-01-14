PYTHONPATH=. python3 tools/infer_det.py \
              -c configs/detection/test/detr_r50.yaml \
              -o global.save_dir=runs \
              global.weights=weights/detr_r50_pt/checkpoints/best.pth 