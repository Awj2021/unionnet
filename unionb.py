import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
from model import make_network
from timm.scheduler import create_scheduler
import utils_matrics, utils
from sklearn.metrics import accuracy_score
import os
import wandb
import ipdb
from torchmetrics.classification import MulticlassCalibrationError

class SuperLayer(nn.Module):
    def __init__(self, args):
        super(SuperLayer, self).__init__()
        self.args = args
        self.model = make_network(args)
        self.device = torch.device(self.args.device)
        self.learning_rate = self.args.lr
        self.dataset_name = self.args.dataset
        self.num_classes = self.args.num_classes
        self.expert_num = self.args.expert_num
        
        self.batch_size = self.args.batch_size

        self.model.add_module("super",
                              nn.Linear(self.num_classes, self.expert_num * self.num_classes, bias=False))
        self.weights_init()
        self.model.to(self.device)
        self.eye = torch.eye(self.num_classes).to(self.device)

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

        for i, (img, eps) in enumerate(train_loader):
            # ipdb.set_trace()
            eps = F.one_hot(eps, num_classes=self.num_classes).float()
            # ipdb.set_trace()
            ep = eps.to(self.device)  # ep is the annotators' labels.
            img = img.to(self.device)

            loss = self.train_batch_new(img, ep)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss
            loss_iter = loss.item()
            print('Epoch: {} | Iter: {} | Loss: {:.4f}'.format(epoch, str((i)), loss_iter))
            wandb.log({"epoch": epoch, "train_iter_loss": loss_iter})
        avg_loss = total_loss / len(train_loader)
        wandb.log({"epoch": epoch, "train_avg_loss": avg_loss})
        if self.args.save_checkpoint:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            if self.args.data_aug:
                # dataset_setting_name = self.args.aug_data_dir.basename().split('.')[0]
                dataset_setting_name = os.path.basename(self.args.aug_data_dir).split('.')[0]
            else:
                dataset_setting_name = 'without_data_aug'
            checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.dataset, dataset_setting_name)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'unionb_checkpoint_{}.pth'.format(epoch)))

        self.lr_scheduler.step(epoch)
        print('*' * 50)
        print('Epoch: {} | total_loss: {:.4f}'.format(epoch, avg_loss))
        print('*' * 50)


    def train_batch_new(self, images, ep):
        y_hat = self.model(images)
        # mid_out = F.softmax(outputs, dim=1)  # y_hat
        theta = F.softmax(self.model.super(self.eye), dim=1)
        out = torch.matmul(y_hat, theta)
        label_one_hot_all = ep.reshape(-1, self.expert_num * self.num_classes) # ep: the annotators labels
        label_all_soft = label_one_hot_all / label_one_hot_all.sum(dim=1, keepdim=True)
        loss = torch.mean(torch.sum(- label_all_soft * torch.log(out), 1))

        return loss

    def save_matrix(self):
        theta = F.softmax(self.model.super(self.eye), dim=1)
        matrix = theta.cpu().data.numpy().transpose(1, 0).reshape(self.expert_num, self.num_classes,
                                                                  self.num_classes)
        np.save("./Method_B_matrix.npy", matrix)

    def weights_init(self):
        """Initialization of the Transition Matrix T"""
        epsilon = 0.00001
        if self.dataset_name == "chaoyang" or self.dataset_name == "cifar10n":
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

        self.model.eval()
        y_hat = None
        loss_hat = 0

        def hook_fn(m, i, o):
            nonlocal y_hat
            y_hat = o.detach()

        total_accuracy = 0
        total_samples = 0

        for batch_idx, (img, gt_label) in enumerate(test_loader):
            img = img.to(self.device)
            gt_label = gt_label.to(self.device)  # y_hat

            hook = self.fc_layer.register_forward_hook(hook_fn)
            y_tilde = self.model(img)  # y_tilde

            loss_y_hat = criterion(y_hat, gt_label)

            loss_hat += loss_y_hat.item()

            acc1 = accuracy_score(y_hat.argmax(-1).cpu(), gt_label.cpu())
            total_accuracy += acc1 * img.shape[0]
            total_samples += img.shape[0]
            print('Iter: {} / {}  Acc: {:.3f}'.format(batch_idx, len(test_loader), acc1))
            metrics = MulticlassCalibrationError(num_classes=self.num_classes, n_bins=5, norm='l1')
            metrics.update(y_hat.float(), gt_label)
            # ipdb.set_trace()
            hook.remove()
        # Get the calibration error
        calibration_error = metrics.compute()
        # fig_, ax_ = metrics.plot()        
        # fig_.savefig(f"./calibration_{epoch}.png")
        avg_loss_hat = loss_hat / len(test_loader)
        avg_accuracy = total_accuracy / total_samples
        wandb.log({"epoch": epoch, "val_avg_loss": avg_loss_hat, "val_avg_accuracy": avg_accuracy, "calibration_error": calibration_error})
        # wandb.log({"epoch": epoch, "val_avg_loss": avg_loss_hat, "val_avg_accuracy": avg_accuracy})
        print(f'Epoch : {epoch}  Average y_hat loss: {avg_loss_hat}')
        print(f'Epoch : {epoch}  Average Accuracy: {avg_accuracy}')
