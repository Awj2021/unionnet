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
            label_1 = []
            label_2 = []
            label_3 = []
            json_path = os.path.join(self.data_path, 'json', 'train.json')
            with open(json_path, 'r') as f:
                load_list = json.load(f)
                for i in range(len(load_list)):
                    img_path = os.path.join(self.data_path, load_list[i]["name"])
                    imgs.append(img_path)
                    labels.append(load_list[i]["label"])
                    label_1.append(load_list[i]["label_A"])
                    label_2.append(load_list[i]["label_B"])
                    label_3.append(load_list[i]["label_C"])

            self.train_data, self.train_labels = np.array(imgs), np.array(labels)

            self.eps = torch.cat((F.one_hot(torch.tensor(label_1), num_classes=args.num_classes).unsqueeze(1),
                                  F.one_hot(torch.tensor(label_2), num_classes=args.num_classes).unsqueeze(1),
                                  F.one_hot(torch.tensor(label_3), num_classes=args.num_classes).unsqueeze(1)), dim=1)
        else:
            imgs = []
            labels = []
            label_1 = []
            label_2 = []
            label_3 = []
            json_path = os.path.join(self.data_path, 'json', 'test.json')
            with open(json_path, 'r') as f:
                load_list = json.load(f)
                for i in range(len(load_list)):
                    img_path = os.path.join(self.data_path, load_list[i]["name"])
                    imgs.append(img_path)
                    labels.append(load_list[i]["label"])
                    label_1.append(load_list[i]["label_A"])
                    label_2.append(load_list[i]["label_B"])
                    label_3.append(load_list[i]["label_C"])
            self.test_data, self.test_labels = np.array(imgs), np.array(labels)

            self.eps = torch.cat((F.one_hot(torch.tensor(label_1), num_classes=args.num_classes).unsqueeze(1),
                                  F.one_hot(torch.tensor(label_2), num_classes=args.num_classes).unsqueeze(1),
                                  F.one_hot(torch.tensor(label_3), num_classes=args.num_classes).unsqueeze(1)), dim=1)

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
            eps = self.eps[item]
            return img, gt_label, eps
            # return img, gt_label

    def __len__(self):
        if self.is_train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def build_dataset(is_train, args):
    if args.dataset == 'Chaoyang':
        dataset = Chaoyang(is_train, args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        raise ValueError('Please input the right name of dataset!!!')

    return dataloader

