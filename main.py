import argparse
from unionb import SuperLayer
from datasets import build_dataset
import os


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
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument('--dataset', type=str, default='Chaoyang', help='Dataset Name')

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

    return parser


def main():
    args = get_args_parser().parse_args()

    print(args)
    print("Start Training...")

    if not os.path.exists(args.checkpoint_dir):
        # Create the directory
        os.makedirs(args.checkpoint_dir)
        print('== Creating the paths for saving the checkpoints...')

    train_dataloader = build_dataset(is_train=True, args=args)
    test_dataloader = build_dataset(is_train=False, args=args)

    model = SuperLayer(args)

    for epoch in range(args.start_epoch, args.epochs):
        model.train_one_epoch(train_dataloader, epoch)
        model.val(test_dataloader, epoch)

    model.save_matrix()


if __name__ == '__main__':
    main()
