import torch
import torch.utils
from tqdm import tqdm
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
from PIL import Image
import torch.nn.functional as F
from torchvision.datasets import CIFAR100

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class Chaoyang(Dataset):
    def __init__(self, is_train, args):
        self.is_train = is_train
        self.class_num = args.num_classes
        self.expert_num = args.expert_num
        self.data_path = args.data_path
        # The input Image should be resized into proper ratio.
        self.train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if self.is_train:  # is_train = True => Train.
            imgs = []
            labels = []
            # label_1 = []
            label_2 = []
            label_3 = []
            json_path = os.path.join(self.data_path, 'json', 'train_label.json')
            with open(json_path, 'r') as f:
                load_list = json.load(f)
                for i in range(len(load_list)):
                    img_path = os.path.join(self.data_path, load_list[i]["name"])
                    imgs.append(img_path)
                    labels.append(load_list[i]["label"])
                    # label_1.append(load_list[i]["label_A"])
                    label_2.append(load_list[i]["label_B"])
                    label_3.append(load_list[i]["label_C"])

            self.train_data, self.train_labels = np.array(imgs), np.array(labels)

            # self.eps = torch.cat((F.one_hot(torch.tensor(label_1), num_classes=args.num_classes).unsqueeze(1),
            #                       F.one_hot(torch.tensor(label_2), num_classes=args.num_classes).unsqueeze(1),
            #                       F.one_hot(torch.tensor(label_3), num_classes=args.num_classes).unsqueeze(1)), dim=1)
            self.eps = torch.cat((F.one_hot(torch.tensor(label_2), num_classes=args.num_classes).unsqueeze(1),
                                  F.one_hot(torch.tensor(label_3), num_classes=args.num_classes).unsqueeze(1)), dim=1)
        else:
            imgs = []
            labels = []
            # label_1 = []
            # label_2 = []
            # label_3 = []
            json_path = os.path.join(self.data_path, 'json', 'test_ori.json')
            with open(json_path, 'r') as f:
                load_list = json.load(f)
                for i in range(len(load_list)):
                    img_path = os.path.join(self.data_path, load_list[i]["name"])
                    imgs.append(img_path)
                    labels.append(load_list[i]["label"])
                    # label_1.append(load_list[i]["label_A"])
                    # label_2.append(load_list[i]["label_B"])
                    # label_3.append(load_list[i]["label_C"])
            self.test_data, self.test_labels = np.array(imgs), np.array(labels)

            # self.eps = torch.cat((F.one_hot(torch.tensor(label_1), num_classes=args.num_classes).unsqueeze(1),
            #                       F.one_hot(torch.tensor(label_2), num_classes=args.num_classes).unsqueeze(1),
            #                       F.one_hot(torch.tensor(label_3), num_classes=args.num_classes).unsqueeze(1)), dim=1)

    def __getitem__(self, item):
        if self.is_train:
            img = self.train_data[item]
            img = Image.open(img).convert('RGB')
            img = self.train_transform(img)

            gt_label = self.train_labels[item]
            eps = self.eps[item]
            return img, gt_label, eps
        else:
            img, gt_label = self.test_data[item], self.test_labels[item]
            img = Image.open(img).convert('RGB')
            img = self.test_transform(img)
            # eps = self.eps[item]
            return img, gt_label
            # return img, gt_label

    def __len__(self):
        if self.is_train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class CIFAR100_IDN(Dataset):
    def __init__(self, is_train, args):
        self.is_train = is_train
        self.class_num = args.num_classes
        self.data_path = args.data_path   # noisy labels path
        self.expert_num = args.expert_num
        self.noise_file = args.noise_file

        if not os.path.exists(self.data_path):
            os.system('wget -O %s/cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'%self.data_path)

        self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
            ])
        self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
            ])
        
        self.noise_data = torch.load(os.path.join(self.data_path, self.noise_file))

        if self.is_train:
            train_dic = unpickle(os.path.join(self.data_path, 'train'))
            self.train_data = train_dic['data']
            self.train_clean_labels = train_dic['fine_labels']
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

            one_hot_labels = []
            for i in range(1, self.expert_num + 1):
                key = f'random_label{i}'
                if key in self.noise_data:
                    one_hot_labels.append(F.one_hot(torch.tensor(self.noise_data[key]), num_classes=args.num_classes).unsqueeze(1))
                else:
                    raise ValueError(f'Please check the key {key} in the noise file!!!')
            self.eps = torch.cat(one_hot_labels, dim=1)
        else:
            test_dic = unpickle(os.path.join(self.data_path, 'test'))
            self.test_data = test_dic['data']
            self.test_labels = test_dic['fine_labels']
            self.test_data = self.test_data.reshape((-1, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))
            # one_hot_labels = []
            # for i in range(1, self.expert_num + 1):
            #     key = f'random_label{i}'
            #     if key in self.noise_data:
            #         one_hot_labels.append(F.one_hot(torch.tensor(self.noise_data[key]), num_classes=args.num_classes).unsqueeze(1))
            #     else:
            #         raise ValueError(f'Key {key} not found in noise_data')

            # self.eps = torch.cat(one_hot_labels, dim=1)

    def __getitem__(self, item):
        if self.is_train:
            img = self.train_data[item]
            # img = Image.open(img).convert('RGB')
            img = Image.fromarray(img)
            img = self.train_transform(img)

            gt_label = self.train_clean_labels[item]
            eps = self.eps[item]
            return img, gt_label, eps
        else:
            img, gt_label = self.test_data[item], self.test_labels[item]
            # img = Image.open(img).convert('RGB')
            img = Image.fromarray(img)
            img = self.test_transform(img)
            # eps = self.eps[item]
            return img, gt_label

    def __len__(self):
        if self.is_train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def build_dataset(is_train, args):
    if args.dataset == 'CIFAR100-IDN':
        dataset = CIFAR100_IDN(is_train, args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    else:
        raise ValueError('Please input the right name of dataset!!!')

    return dataloader
def build_dataset(is_train, args):
    if args.dataset == 'Chaoyang':
        dataset = Chaoyang(is_train, args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    elif args.dataset == 'cifar100':
        dataset = CIFAR100_IDN(is_train, args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        raise ValueError('Please input the right name of dataset!!!')

    return dataloader

