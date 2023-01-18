MODEL_NAME=$1
FOLD_ID=$2
EXP_NAME="kvasir_${MODEL_NAME}_fold${FOLD_ID}"
PYTHONPATH=. python3 tools/train_cls.py \
              -c configs/classification/${MODEL_NAME}.yaml \
              -o global.save_dir=runs \
              global.exp_name=$EXP_NAME \
              global.exist_ok=False \
              data.dataset.train.args.csv_path=data/kvasir/folds/train_fold${FOLD_ID}.csv \
              data.dataset.val.args.csv_path=data/kvasir/folds/val_fold${FOLD_ID}.csv