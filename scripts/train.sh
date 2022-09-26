EXP_NAME=$1
PYTHONPATH=. python tools/train.py \
              -c configs/classification/pipeline.yaml \
              -o trainer.args.num_iterations=300000 \
              trainer.args.print_interval=100 \
              global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=True