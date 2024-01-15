import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
from model import make_network
from timm.scheduler import create_scheduler
import utils
from sklearn.metrics import accuracy_score
import os


class SuperLayer(nn.Module):
    def __init__(self, args):
        super(SuperLayer, self).__init__()
        self.args = args
        self.model = make_network(args)
        self.num_classes = self.args.num_classes
        self.device = torch.device(self.args.device)
        self.learning_rate = self.args.lr
        self.expert_num = self.args.expert_num
        self.dataset_name = self.args.dataset
        self.batch_size = self.args.batch_size

        self.model.add_module("super",
                              nn.Linear(self.num_classes, self.expert_num * self.num_classes, bias=False))
        self.weights_init()
        self.model.to(self.device)
        self.eye = torch.eye(self.num_classes).to(self.device)

        # TODO: Add one lr_scheduler.
        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError("Please Check the optimizer for training...")

        self.lr_scheduler, _ = create_scheduler(self.args, self.optimizer)

        self.fc_layer = self.model.fc

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, (img, gt_label, eps) in enumerate(train_loader):

            ep = eps.to(self.device)  # ep is the annotators' labels.
            img = img.to(self.device)

            loss = self.train_batch_new(img, ep)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # TODO: add the lr_scheduler.
            total_loss += loss

        if self.args.save_checkpoint:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(self.args.checkpoint_dir, 'unionb_checkpoint_{}.pth'.format(epoch)))

        self.lr_scheduler.step(epoch)
        print('Epoch: {} | total_loss: {:.4f}'.format(epoch, total_loss))

    def train_batch_new(self, images, ep):
        y_hat = self.model(images)
        # mid_out = F.softmax(outputs, dim=1)  # y_hat
        theta = F.softmax(self.model.super(self.eye), dim=1)
        out = torch.matmul(y_hat, theta)

        label_one_hot_all = ep.reshape(-1, self.expert_num * self.num_classes) # ep: the annotators labels
        label_all_soft = label_one_hot_all / label_one_hot_all.sum(dim=1, keepdim=True)
        loss = torch.mean(torch.sum(- label_all_soft * torch.log(out), 1))
        # loss = kl_loss_function(out, label_one_hot_all, self.priori.cpu())
        return loss

    def save_matrix(self):
        theta = F.softmax(self.model.super(self.eye), dim=1)
        matrix = theta.cpu().data.numpy().transpose(1, 0).reshape(self.expert_num, self.num_classes,
                                                                  self.num_classes)
        np.save("./Method_B_matrix.npy", matrix)

    def weights_init(self):
        """Initialization of the Transition Matrix T"""
        epsilon = 0.00001
        if self.dataset_name == "Chaoyang":
            theta = (1 - epsilon) * torch.eye(self.num_classes) + epsilon / (self.num_classes - 1) * (
                    1 - torch.eye(self.num_classes))
            self.model.super.weight.data = theta.repeat(self.expert_num, 1)
        else:
            raise ValueError("please check the dataset_name again")

    # In fact, we just would like to get the target labels.
    # we thought that the generated labels would be better than the majority-voted labels provided by original dataset.

    @torch.no_grad()
    def val(self, test_loader, epoch):
        criterion = torch.nn.CrossEntropyLoss()
        metric_logger = utils.MetricLogger(delimiter="  ")

        self.model.eval()
        y_hat = None
        loss_hat = 0

        def hook_fn(m, i, o):
            nonlocal y_hat
            y_hat = o.detach()

        for batch_idx, (img, gt_label, eps) in enumerate(test_loader):
            img = img.to(self.device)
            gt_label = gt_label.to(self.device)  # y_hat

            hook = self.fc_layer.register_forward_hook(hook_fn)
            y_tilde = self.model(img)  # y_tilde

            loss_y_hat = criterion(y_hat, gt_label)

            loss_hat += loss_y_hat.item()
            if self.num_classes < 5:
                # ipdb.set_trace()
                # self.accuracy.update(y_hat.argmax(-1), gt_label)
                acc1 = accuracy_score(y_hat.argmax(-1).cpu(), gt_label.cpu())
                print('Iter: {} / {}  Acc: {:.3f}'.format(batch_idx, len(test_loader), acc1))
                metric_logger.meters['acc1'].update(acc1, n=self.batch_size)
            hook.remove()

        avg_loss_hat = loss_hat / len(test_loader.dataset)
        print(f'Epoch : {epoch}  Average y_hat loss: {avg_loss_hat}')
        print(f'Epoch : {epoch}, Average Accuracy: {metric_logger.acc1}')
