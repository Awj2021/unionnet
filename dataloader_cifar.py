from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10  # setting the clean cifar10 dataset.
import torchvision.datasets as datasets
from cifar import CIFAR10_MR # just setting the multi rater dataset.

def load_mr_data(batchsize:int, numworkers:int, args):
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_train = CIFAR10_MR(
                        root = './',
                        train = True,
                        download = False,
                        noise_type = args.noisy_type,
                        noise_path = args.noise_path,
                        transform = trans,
                        data_aug=args.data_aug,
                        aug_data_dir=args.aug_data_dir,
                    )
    trainloader = DataLoader(
                        data_train,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        shuffle=True,
                        drop_last = True
                    )
    return trainloader


def load_clean_test_data(batchsize:int, numworkers:int) -> DataLoader:
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_test = CIFAR10(
                        root = './',
                        train = False,
                        download = True,
                        transform=trans
                        )
    testloader = DataLoader(
                        data_test,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        drop_last = False
                    )
    return testloader

def transback(data:Tensor) -> Tensor:
    return data / 2 + 0.5
