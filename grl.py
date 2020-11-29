import copy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from utils.Fed import FedAvg
from utils.options import args_parser
from utils.sampling import data_iid, data_noniid, load_data, load_unlabeled_data, calulate_non_iidness
from model import models
from model.Update import LocalUpdate, DatasetSplit

'''
Baseline for semi-supervised learning
client performs consistency regulation
server conducts supervised learning
there is GRL layer for domain adaptation
'''

def client_train(args, unlabeled_weak_data, unlabeled_strong_data, model, idxs):
    target_train_weak_loader = DataLoader(DatasetSplit(unlabeled_weak_data, idxs), batch_size=args.local_bs,
                                          shuffle=True, num_workers=32)
    target_train_strong_loader = DataLoader(DatasetSplit(unlabeled_strong_data, idxs), batch_size=args.local_bs,
                                            shuffle=True, num_workers=32)

    optimizer = torch.optim.SGD(
        [{'params': model.base_network.parameters()},
         {'params': model.classifier_layer.parameters(),
          'lr': 10 * args.lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.l2_decay)

    model.train()
    
    for e in range(args.local_ep):
        
        n_batch = len(target_train_weak_loader)

        for i in range(n_batch):
            weak_data, _ = iter(target_train_weak_loader).next()
            strong_data,_ = iter(target_train_strong_loader).next()
            weak_data, strong_data = weak_data.to(args.device), strong_data.to(args.device)
            domain_label = torch.empty(len(weak_data)).fill_(1).to(args.device).long()

            clf_weak, domain_weak = model(weak_data)
            clf_strong, domain_strong = model(strong_data)

            pseudo_label = torch.softmax(clf_weak.detach_(), dim=-1)
            max_probs, pseudo_u = torch.max(pseudo_label, dim=-1)

            # print(max_probs)
            # mask = max_probs.ge(0.05)
            # clf_strong = clf_strong[mask]
            # pseudo_u = pseudo_u[mask]

            optimizer.zero_grad()

            loss_class = criterion(clf_strong, pseudo_u)
            domain = (domain_weak+domain_strong)/2
            loss_domain = criterion(domain, domain_label)
            loss = loss_class + loss_domain*args.domain_lamda
            loss.backward()

            optimizer.step()

        print("local_ep ", e)

    return model.state_dict()


def server_train(args, source_loader, model):
    model.train()
    optimizer = torch.optim.SGD(
        [{'params': model.base_network.parameters()},
         {'params': model.classifier_layer.parameters(),
          'lr': 10 * args.lr}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.l2_decay)

    for i, (data, label) in enumerate(source_loader):
        data, label = data.to(args.device), label.to(args.device)
        
        optimizer.zero_grad()

        domain_label = torch.empty(len(label)).fill_(1).to(args.device).long()
        clf, domain = model(data)
        loss_class = criterion(clf, label)
        loss_domain = criterion(domain, domain_label)
        loss = loss_class + loss_domain*args.domain_lamda
        loss.backward()

        optimizer.step()

    w_server = model.state_dict()
    return w_server


def test(model, target_test_loader):
    model.eval()
    correct = 0
    #criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output, domain = model(data)
            #loss = criterion(s_output, target)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

    print('max correct: {}, accuracy{: .2f}%\n'.format(
        correct, 100. * correct / len_target_dataset))
    return 1. * correct / len_target_dataset



if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)
    criterion = torch.nn.CrossEntropyLoss()

    # load dataset
    source_name = "webcam"
    target_name = "amazon"
    print('Src: %s, Tar: %s' % (source_name, target_name))

    source_data, _, target_test_data = load_data(source_name, target_name, data_dir="/data/xian/Office-31/")
    unlabeled_weak_data, unlabeled_strong_data = load_unlabeled_data(target_name, data_dir="/data/xian/Office-31/")

    if args.iid:  # default false
        dict_users, dict_classes = data_iid(unlabeled_weak_data, args.num_users)
    else:
        dict_users, dict_classes = data_noniid(unlabeled_weak_data, args.num_users)

    # calculate non-iidness
    non_iidness = calulate_non_iidness(args, dict_classes)
    print(f"non_iidness = {non_iidness}")

    source_loader = DataLoader(source_data, batch_size=args.local_bs, shuffle=True, num_workers=8)
    target_test_loader = DataLoader(target_test_data, batch_size=args.bs, shuffle=True, num_workers=8)

    # define model
    global_model = models.GRL_Net(args.n_class).to(args.device)

    with SummaryWriter() as writer:
        for epoch in range(args.epochs):

            # client train, unsupervised training
            w_locals = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            j=0
            for idx in idxs_users:
                j = j + 1
                print("user ", j)
                ##client training
                w_local = client_train(args, unlabeled_weak_data, unlabeled_strong_data, copy.deepcopy(global_model).to(args.device), idxs=dict_users[idx])
                w_locals.append(copy.deepcopy(w_local))

            #server training
            w_server = server_train(args, source_loader, copy.deepcopy(global_model).to(args.device))

            # parameter aggregation
            w_locals.append(copy.deepcopy(w_server))
            w_avg = FedAvg(w_locals)

            global_model.load_state_dict(w_avg)
            print("global epoch ", epoch)

            # evaluation and save
            if epoch % args.eval_interval == 0:
                # evaluate
                acc = test(global_model.to(args.device), target_test_loader)
                writer.add_scalar("accuracy", acc, global_step=epoch)