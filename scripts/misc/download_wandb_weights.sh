%cd /home/htluc/vocal-folds/tools/
PYTHONPATH=. python3 download_wandb_weights.py \
--filename "checkpoints/best.pth" \
--run_path "hcmus-dcu/aim/fr9g3812" \
--rename "tf_effnet_b0_fold_0_best.pth"
