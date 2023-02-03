import argparse
from theseus.base.utilities.download import download_from_wandb
from theseus.base.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger("main")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='most of the time is checkpoints/best.pth')
    parser.add_argument('--run_path', type=str, help='model path on WANDB server')
    parser.add_argument('--save_dir', type=str, help='the directory to save the weight', nargs='+', default=".")
    parser.add_argument('--rename',   type=str, help='the new name for the weight')
    opt = parser.parse_args()
    
    download_from_wandb(
        filename=opt.filename,
        run_path=opt.run_path,
        save_dir=opt.save_dir,
        rename=opt.rename
    )