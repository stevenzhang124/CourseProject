import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
from model import models
from model.Update import LocalUpdate
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from utils.Fed import FedAvg
from utils.options import args_parser
from utils.sampling import data_iid, data_noniid, load_data

matplotlib.use('Agg')


# parse args
args = args_parser()
args.device = torch.device('cuda:{}'.format(
    args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
torch.manual_seed(args.seed)

# load dataset
source_name = "webcam"
target_name = "amazon"
print('Src: %s, Tar: %s' % (source_name, target_name))
source_data, target_train_data, target_test_data = load_data(
    source_name, target_name, data_dir='dataset')
source_loader = DataLoader(
    source_data, batch_size=args.local_bs, shuffle=True, num_workers=8)
target_train_loader = DataLoader(
    target_train_data, batch_size=args.local_bs, shuffle=True, num_workers=8)
target_test_loader = DataLoader(
    target_test_data, batch_size=args.bs, shuffle=True, num_workers=8)

best_acc = 0
criterion = torch.nn.CrossEntropyLoss()


def test(model):
    """
    evaluate the accuracy
    """
    correct = 0
    len_target_dataset = len(target_test_loader.dataset)

    with torch.no_grad():
        for data, label in target_test_loader:
            data, label = data.to(args.device), label.to(args.device)
            pred = torch.max(model.predict(data), 1)[1]
            correct += torch.sum(pred == label)

    print('max correct: {}, accuracy{: .2f}%\n'.format(
        correct, 100. * correct / len_target_dataset))
    return 1. * correct / len_target_dataset


def baseline_1():
    """
    train on source domain (labeled). No FL.
    """
    model = models.Transfer_Net(args.n_class, use_domain_loss=False).to(args.device)
    optimizer = torch.optim.SGD(
        [{'params': model.base_network.parameters()},
        {'params': model.fc_layers.parameters(), 'lr': 5 * args.lr},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * args.lr},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * args.lr}], 
        lr=args.lr, momentum=args.momentum, weight_decay=args.l2_decay)

    global best_acc

    with SummaryWriter() as writer:

        for epoch in range(args.epochs):

            model.train()

            for batch_i, (data, label) in enumerate(source_loader):
                batches_done = len(source_loader) * epoch + batch_i
                data, label = data.to(args.device), label.to(args.device)
                clf = model(data, 0)    # 0 is a place holder
                optimizer.zero_grad()
                loss = criterion(clf, label)
                loss.backward()
                optimizer.step()

                writer.add_scalar("train_loss", loss, global_step=batches_done)
                print(f'Epoch {epoch}-Batch {batch_i}, loss: {loss:.3}')

            # evaluate and save
            if epoch % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    # evaluate
                    acc = test(model)
                    writer.add_scalar("accuracy", acc, global_step=epoch)
                    # save last, best and delete
                    ckpt = {'epoch': epoch, 'model': model.state_dict()}
                    if acc > best_acc and epoch < args.epochs - 1:
                        torch.save(ckpt, f"./save/best_baseline1.pt")
                        best_acc = acc
                    del ckpt


def baseline_2():
    """
    train on source domain (labeld) + target domain (unlabeled). No FL.
    """
    model = models.Transfer_Net(
        args.n_class, transfer_loss='mmd', base_net='resnet18').to(args.device)
    optimizer = torch.optim.SGD(
        [{'params': model.base_network.parameters()},
        {'params': model.fc_layers.parameters(), 'lr': 5 * args.lr},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * args.lr},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * args.lr}], 
        lr=args.lr, momentum=args.momentum, weight_decay=args.l2_decay)

    global best_acc

    with SummaryWriter() as writer:

        for epoch in range(args.epochs):

            model.train()

            for batch_i, ((data_source, label_source), (data_target, _)) in enumerate(zip(source_loader, target_train_loader)):

                batches_done = len(source_loader) * epoch + batch_i
                data_source, label_source = data_source.to(
                    args.device), label_source.to(args.device)
                data_target = data_target.to(args.device)

                optimizer.zero_grad()
                label_source_pred, transfer_loss = model(
                    data_source, data_target)
                clf_loss = criterion(label_source_pred, label_source)
                loss = clf_loss + args.lam * transfer_loss
                loss.backward()
                optimizer.step()

                writer.add_scalar("train_loss", loss, global_step=batches_done)
                print(f'Epoch {epoch}-Batch {batch_i}, loss: {loss:.3}')

            # evaluate and save
            if epoch % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    # evaluate
                    acc = test(model)
                    writer.add_scalar("accuracy", acc, global_step=epoch)
                    # save last, best and delete
                    ckpt = {'epoch': epoch, 'model': model.state_dict()}
                    if acc > best_acc and epoch < args.epochs - 1:
                        torch.save(ckpt, f"./save/best_baseline2.pt")
                        best_acc = acc
                    del ckpt


if __name__ == '__main__':

    # baseline_1()
    baseline_2()
