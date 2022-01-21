from tools.utils.getter import *
import argparse
import os

parser = argparse.ArgumentParser('Training Classification')
parser.add_argument('--config', type=str, default="./configs/configs.yaml", help='Config file')
parser.add_argument('--print_per_iter', type=int, default=300, help='Number of iteration to print')
parser.add_argument('--val_interval', type=int, default=2, help='Number of epoches between valing phases')
parser.add_argument('--gradcam_visualization', action='store_true', help='whether to visualize box to ./sample when validating (for debug), default=off')
parser.add_argument('--save_interval', type=int, default=1000, help='Number of steps between saving')
parser.add_argument('--resume', type=str, default=None,
                    help='whether to load weights from a checkpoint, set None to initialize')
parser.add_argument('--saved_path', type=str, default='./weights')
parser.add_argument('--freeze_backbone', action='store_true', help='whether to freeze the backbone')
  

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    devices_info = get_devices_info(config.gpu_devices)
    
    trainset, valset, trainloader, valloader = get_dataset_and_dataloader(config)
  
    net = get_model(config, trainset.num_classes)

    metric = [
        AccuracyMetric(),
        F1ScoreMetric(),
        BalancedAccuracyMetric(num_classes=trainset.num_classes), 
        ConfusionMatrix(trainset.classes), 
    ]

    criterion = get_loss(config.loss_fn, num_classes=trainset.num_classes)
    optimizer, optimizer_params = get_lr_policy(config.lr_policy)

    model = Classifier(
            model = net,
            metrics=metric,
            scaler=NativeScaler(),
            criterion=criterion,
            optimizer= optimizer,
            optim_params = optimizer_params,     
            device = device)

    if args.resume is not None:                
        load_checkpoint(model, args.resume)
        start_epoch, start_iter, best_value = get_epoch_iters(args.resume)
    else:
        print('Not resume. Initialize weights')
        start_epoch, start_iter, best_value = 0, 0, 0.0
        
    scheduler, step_per_epoch = get_lr_scheduler(
        model.optimizer, 
        train_len=len(trainloader),
        lr_config=config.lr_scheduler,
        num_epochs=config.num_epochs)

    if args.resume is not None:                 
        old_log = find_old_log(args.resume)
    else:
        old_log = None

    args.saved_path = os.path.join(
        args.saved_path, 
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    trainer = Trainer(config,
                     model,
                     trainloader, 
                     valloader,
                     checkpoint = Checkpoint(save_per_iter=args.save_interval, path = args.saved_path),
                     best_value=best_value,
                     logger = Logger(log_dir=args.saved_path, resume=old_log),
                     scheduler = scheduler,
                     visualize_when_val = args.gradcam_visualization,
                     evaluate_per_epoch = args.val_interval,
                     step_per_epoch = step_per_epoch)
    
    print("##########   DATASET INFO   ##########")
    print("Trainset: ")
    print(trainset)
    print("Valset: ")
    print(valset)
    print()
    print(trainer)
    print()
    print(config)
    print(f'Training with {num_gpus} gpu(s)')
    print(devices_info)
    print(f"Start training at [{start_epoch}|{start_iter}]")
    print(f"Current best F1: {best_value}")
    
    trainer.fit(start_epoch = start_epoch, start_iter = start_iter, num_epochs=config.num_epochs, print_per_iter=args.print_per_iter)

    

if __name__ == '__main__':
     
    args = parser.parse_args()
    config = Config(args.config)

    train(args, config)
    


