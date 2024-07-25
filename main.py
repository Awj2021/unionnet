import argparse
from unionb import SuperLayer
from datasets import build_dataset
from dataloader_cifar import load_mr_data, load_clean_test_data
import os
import yaml
import ipdb
import wandb


def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)
    parser.add_argument('--expert_num', type=int, default=3, help='Number of experts')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of Classes')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--data_path', type=str, default='/scratch/projects/multirater/chaoyang/',
                                                   help='path of dataset.')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start Epoch')
    parser.add_argument('--epochs', type=int, default=100, help='epoch numbers of training')
    parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate')
    parser.add_argument('--network', type=str, default='resnet50', help='Type of network.')
    parser.add_argument('--pretrained', type=bool, default=False, help='Whether to use the pretrained model.')

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument('--dataset', type=str, choices=['chaoyang', 'cifar10n', 'cifar100n'], default='cifar10n', help='Dataset Name')

    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training.')

    # Optimizer, Following the PVT (Pyramid Transformer Network.) settting.
    parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--save_checkpoint', type=bool, default=True, help='Save the checkpoint...')
    parser.add_argument('--checkpoint_dir', type=str, default='./models/unionb', help='The dir for saving checkpoint.')

    # set the parameters for the dataset.
    parser.add_argument('--noisy_type', type=str, default='multi_rater', choices=['clean', 'multi_rater'], help='Type of noise')
    parser.add_argument('--data_aug', type=bool, default=False, help='Whether to use data augmentation')
    parser.add_argument('--aug_data_dir', type=str, default='./cifar-10-batches-py/gen_samples_and_lab_disagree_x1.pt', 
                        help='Path of data augmentation files.')
    parser.add_argument('--noise_path', type=str, default='./cifar-10-batches-py/CIFAR-10_human.pt',
                        help='the multi-rater noise data path.')
    parser.add_argument('--config_file', type=str, default='./configs/cifar10n.yaml', help='The config file to set the specific parameters.')
    return parser


def main():
    args = get_args_parser().parse_args()
    args_dict = vars(args)

    # add the config files. And write the parameters to the config files.
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    args_dict.update(config)
    args = argparse.Namespace(**args_dict)
    wandb.init(project='unionb', config=args)

    print("Start Training...")

    if not os.path.exists(os.path.join(args.checkpoint_dir, args.dataset)):
        # Create the directory
        os.makedirs(os.path.join(args.checkpoint_dir, args.dataset))
        print('== Creating the paths for saving the checkpoints...')

    if args.dataset == 'chaoyang':
        train_dataloader = build_dataset(is_train=True, args=args)
        test_dataloader = build_dataset(is_train=False, args=args)

    elif args.dataset == 'cifar10n':
        train_dataloader = load_mr_data(args.batch_size, 4, args)
        test_dataloader = load_clean_test_data(args.batch_size, 4)

    model = SuperLayer(args)

    for epoch in range(args.start_epoch, args.epochs):
        model.train_one_epoch(train_dataloader, epoch)
        model.val(test_dataloader, epoch)

    model.save_matrix()


if __name__ == '__main__':
    main()
