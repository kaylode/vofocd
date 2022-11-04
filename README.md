# Vocal Fold Classification

## **Installation**

```
git clone https://github.com/kaylode/vocal-fold-isbi.git
cd vocal-fold-isbi
pip install -e .
```

## **Reproduction**

- Build docker image
```
DOCKER_BUILDKIT=1 docker build -t vocalfold:latest .
```

- Run docker image
```
docker run -it --rm --gpus '"device=0"' --name vocalfold -v $(pwd):/workspace vocalfold
```

## **Dataset**
Download data:
```
sh scripts/download_data.sh
```

## **Training & Testing**

Run train scripts:

- Raw images --> convnext_small: `sh scripts/train_small_only.sh`
- Raw images --> convnext_nano: `sh scripts/train_nano_only.sh`
- Raw images --> unet --> convnext_nano: `sh scripts/train_nano_unet.sh`
- Raw images --> medtex (small+nano): `sh scripts/train_medtex_basic.sh`
- Raw images (teacher) + Ersgan (student) --> medtex (small+nano): `sh scripts/train_medtex_ersgan_1.sh`
- Raw images (teacher, student) + Ersgan (student) --> medtex (small+nano): Upcoming


Run eval script:
```
sh scripts/eval.sh {run_name} {config_path} {weight}
```


## **Resources**

Template: https://github.com/kaylode/theseus.git
