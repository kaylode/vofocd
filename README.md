# Vocal Folds

## **Installation**

```
git clone https://github.com/kaylode/vocal-folds.git
cd vocal-folds
pip install -r requirements.txt
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

## Tasks

- [README: Vocal fold classification](./docs/classification.md)
- [README: Vocal fold detection](./docs/detection.md)



## **Resources**

For most of the documentation, please read source code at: https://github.com/kaylode/theseus.git