from tools.utils.getter import *
import argparse
import os

parser = argparse.ArgumentParser('Evaluate classification')
parser.add_argument('--weight', type=str, help='checkpoint file')


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
seed_everything()

def train(args, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    num_gpus = len(config.gpu_devices.split(','))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    devices_info = get_devices_info(config.gpu_devices)
    
    trainset, valset, trainloader, valloader = get_dataset_and_dataloader(config)
  
    net = BaseTimmModel(
        name=config.model_name, 
        num_classes=trainset.num_classes)

    metric = [
        AccuracyMetric(),
        F1ScoreMetric(),
        BalancedAccuracyMetric(num_classes=trainset.num_classes), 
        ConfusionMatrix(trainset.classes), 
    ]

    criterion = get_loss(config.loss_fn, num_classes=trainset.num_classes)

    model = Classifier(
            model = net,
            metrics=metric, 
            criterion=criterion,
            device = device)

    if args.weight is not None:                
        load_checkpoint(model, args.weight)

    trainer = Trainer(config,
                     model,
                     trainloader, 
                     valloader,
                     visualize_when_val = False)

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
  
    trainer.evaluate_epoch()

if __name__ == '__main__':
    
    args = parser.parse_args()
    config = get_config(args.weight)

    train(args, config)
    


