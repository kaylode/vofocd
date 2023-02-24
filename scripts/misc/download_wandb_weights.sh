%cd /home/htluc/vocal-folds/tools/
PYTHONPATH=. python3 download_wandb_weights.py \
--filename "checkpoints/best.pth" \
--run_path "hcmus-dcu/aim/fr9g3812" \
<<<<<<< HEAD
--save_dir /mnt/4TBSSD/pmkhoi/isbi/weights/picoconvnext_vocal_0

=======
--rename "tf_effnet_b0_fold_0_best.pth"
>>>>>>> b7846f86f2f6d6e3de1e475392157b6b215164d5
