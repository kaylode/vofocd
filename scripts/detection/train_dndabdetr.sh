MODEL_NAME=$1
FOLD_ID=$2
EXP_NAME=dndabdetr_${MODEL_NAME}_fold${FOLD_ID}_test
PYTHONPATH=. python3 tools/detection/train_det.py \
              -c /mnt/4TBSSD/pmkhoi/isbi/configs/detection/models/dn_dab_detr_custom.yaml \
              -o global.exp_name=$EXP_NAME \
              global.exist_ok=True \
              global.pretrained=/mnt/4TBSSD/pmkhoi/isbi/weights/DN-DAB-DETR-R50-DC5.pth \
              model.args.backbone_name=${MODEL_NAME} \
              data.dataset.train.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_train.json \
              data.dataset.val.args.label_path=data/aim/annotations/annotation_${FOLD_ID}_val.json