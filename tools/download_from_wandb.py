from theseus.base.utilities.download import download_from_wandb

run_id = '8ll617vw'

download_from_wandb(
    filename='checkpoints/best.pth',
    run_path=f'hcmus-dcu/aim/{run_id}',
    save_dir=f'weights/{run_id}',
)

download_from_wandb(
    filename='pipeline.yaml',
    run_path=f'hcmus-dcu/aim/{run_id}',
    save_dir=f'weights/{run_id}',
)


