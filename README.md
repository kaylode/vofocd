# Vocal Fold Classification


## Dataset
https://drive.google.com/drive/folders/1oRrHrfBML67z9VR1uhR0roDObtKM7-Lx?usp=sharing

## Installation

```
git clone https://github.com/kaylode/vocal-fold-isbi.git
cd vocal-fold-isbi
pip install -e .
```

## Reproduction

- Build docker image
```
DOCKER_BUILDKIT=1 docker build -t vocalfold:latest .
```

- Run docker image
```
docker run -it --rm --gpus '"device=0"' --name vocalfold -v $(pwd):/workspace vocalfold
```

## Training & Testing

Comming soon


## Resources

Template: https://github.com/kaylode/theseus.git
