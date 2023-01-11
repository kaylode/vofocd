PYTHONPATH=. python3 tools/train.py \
              -c $2 \
              -o global.save_dir=runs \
              global.exp_name=$1 \
              global.exist_ok=True \
              global.pretrained=$3