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

Run train script:
```
sh scripts/train.sh {exp_name} {save_dir}
```

Run eval script:
```
sh scripts/eval.sh {config_path} {weight}
```


## **Resources**

Template: https://github.com/kaylode/theseus.git
