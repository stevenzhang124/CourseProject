import random

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, source, target_train, target_test, idxs=None):
        self.args = args
        self.source_loader = DataLoader(
            source, batch_size=self.args.local_bs, shuffle=True, num_workers=4)
        self.target_train_loader = DataLoader(DatasetSplit(
            target_train, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=4)
        self.target_test_loader = DataLoader(
            target_test, batch_size=self.args.bs, shuffle=True, num_workers=4)

    def train(self, model):
        optimizer = torch.optim.SGD([
            {'params': model.base_network.parameters()},
            {'params': model.bottleneck_layer.parameters(), 'lr': 10 *
             self.args.lr},
            {'params': model.classifier_layer.parameters(), 'lr': 10 *
             self.args.lr},
        ], lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.l2_decay)

        len_source_loader = len(self.source_loader)
        len_target_loader = len(self.target_train_loader)

        epoch_loss = []
        for e in range(self.args.local_ep):
            batch_loss = []
            model.train()
            iter_source, iter_target = iter(self.source_loader), iter(self.target_train_loader)
            n_batch = min(len_source_loader, len_target_loader)
            criterion = torch.nn.CrossEntropyLoss()
            for i in range(n_batch):
                data_source, label_source = iter_source.next()
                data_target, _ = iter_target.next()
                data_source, label_source = data_source.to(
                    self.args.device), label_source.to(self.args.device)
                data_target = data_target.to(self.args.device)

                optimizer.zero_grad()
                label_source_pred, transfer_loss = model(data_source, data_target)
                clf_loss = criterion(label_source_pred, label_source)
                loss = clf_loss + self.args.lam * transfer_loss
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def test(self, model):
        model.eval()
        correct = 0
        #criterion = torch.nn.CrossEntropyLoss()
        len_target_dataset = len(self.target_test_loader.dataset)
        with torch.no_grad():
            for data, target in self.target_test_loader:
                data, target = data.to(
                    self.args.device), target.to(self.args.device)
                s_output = model.predict(data)
                #loss = criterion(s_output, target)
                pred = torch.max(s_output, 1)[1]
                correct += torch.sum(pred == target)

        print('max correct: {}, accuracy{: .2f}%\n'.format(
            correct, 100. * correct / len_target_dataset))
        return 1. * correct / len_target_dataset
