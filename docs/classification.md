# Vocal Fold Classification

## **Dataset**
...

## **Execution**

Run train scripts:

- Raw images --> convnext_small: `sh scripts/classification/train_small_only.sh`
- Raw images --> convnext_nano: `sh scripts/classification/train_nano_only.sh`
- Raw images --> unet --> convnext_nano: `sh scripts/classification/train_nano_unet.sh`
- Raw images --> medtex (small+nano): `sh scripts/classification/train_medtex_basic.sh`
- Raw images (teacher) + Ersgan (student) --> medtex (small+nano): `sh scripts/classification/train_medtex_ersgan_1.sh`
- Raw images (teacher, student) + Ersgan (student) --> medtex (small+nano): Upcoming


Run eval script:
```
sh scripts/classification/eval.sh {run_name} {config_path} {weight}
```


## **Resources**

- Please refer to `https://github.com/kaylode/theseus/tree/v1.1.1/theseus/cv/classification` for the core framework.
