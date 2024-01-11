from alg.classification.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from tqdm import tqdm


class SuperCL(BaseModel):
    def __init__(self, train_dataset, device):
        super(SuperCL, self).__init__(train_dataset, device)
        self.train_dataset = train_dataset

        self.model.add_module("super",
                              nn.Linear(self.num_classes, self.train_dataset.expert_num * self.num_classes, bias=False))
        self.weights_init(expert_tmatrix=train_dataset.expert_tmatrix)

        self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_loader, data_loader_agg, epoch, disable_tqdm=True):
        self.model.train()

        total_loss = 0

        for batch_idx, (ind, left_data, right_data, true_label) in enumerate(tqdm(train_loader, disable=disable_tqdm)):
            if left_data.size()[0] != self.batch_size:
                continue

            ep = Variable(right_data).float().to(self.device)
            true_label = Variable(true_label).long().to(self.device)
            images = Variable(left_data).float().to(self.device)

            loss = self.train_batch_new(ind, images, ep, true_label, epoch)

            # True label
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss

        return total_loss

    def train_batch_new(self, ind, images, ep, true_label, epoch):
        outputs = self.model(images)
        mid_out = F.softmax(outputs, dim=1)

        out = F.softmax(self.model.super(mid_out), dim=1)

        label_one_hot_all = ep.reshape(-1, self.train_dataset.expert_num * self.num_classes)

        loss = torch.mean(torch.sum(- label_one_hot_all * torch.log(out), 1))

        return loss

    def save_matrix(self):
        import numpy as np
        theta = self.model.super.weight.data
        matrix = theta.cpu().data.numpy().reshape(self.train_dataset.expert_num, self.num_classes, self.num_classes)
        np.save("./data_file/labelme/our1_matrix.npy", matrix)

    def weights_init(self, expert_tmatrix):
        epsilon = 0.00001
        if self.train_dataset.dataset_name not in ["cifar10", "mgc"]:
            theta = (1 - epsilon) * torch.eye(self.num_classes) + epsilon / (self.num_classes - 1) * (
                        1 - torch.eye(self.num_classes))
            self.model.super.weight.data = theta.repeat(self.train_dataset.expert_num, 1)
        else:
            theta = expert_tmatrix.reshape(-1, self.num_classes) + epsilon
            self.model.super.weight.data = torch.log(theta)