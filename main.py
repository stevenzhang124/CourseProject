import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
from model import models
from model.Update import LocalUpdate
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from utils.Fed import FedAvg
from utils.options import args_parser
from utils.sampling import data_iid, data_noniid, load_data

matplotlib.use('Agg')


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)

    # load dataset and distribute to users
    source_name = "webcam"
    target_name = "amazon"

    print('Src: %s, Tar: %s' % (source_name, target_name))
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(
        os.path.dirname(current_path) + os.path.sep + ".")
    father_path = "/data/xian/Office-31/"
    source_data, target_train_data, target_test_data = load_data(
        source_name, target_name, data_dir=father_path)

    if args.iid:    # default false
        dict_users = data_iid(target_train_data, args.num_users)
    else:
        dict_users = data_noniid(target_train_data, args.num_users)

    # load global model
    model = models.Transfer_Net(
        args.n_class, transfer_loss='mmd', base_net='resnet18').to(args.device)

    # copy weights
    w_glob = model.state_dict()

    # training
    loss_train = []
    best_acc = 0

    with SummaryWriter() as writer:
        for epoch in range(args.epochs):
            w_locals, loss_locals = [], []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(
                range(args.num_users), m, replace=False)
            j = 0
            for idx in idxs_users:
                local = LocalUpdate(
                    args, source_data, target_train_data, target_test_data, epoch, dict_users[idx])
                j = j + 1
                print(j)
                w, loss = local.train(copy.deepcopy(model).to(args.device))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))

            # update global weights
            w_glob = FedAvg(w_locals)

            # copy weight to global model
            model.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
            loss_train.append(loss_avg)
            writer.add_scalar("loss_avg", loss_avg, global_step=epoch)

            # evaluation and save
            if epoch % args.eval_interval == 0:
                # evaluate
                acc = local.test(model.to(args.device))
                writer.add_scalar("accuracy", acc, global_step=epoch)
                # save last, best and delete
                ckpt = {'epoch': epoch, 'model': model.state_dict()}
                if acc > best_acc and epoch < args.epochs - 1:
                    torch.save(ckpt, f"./save/best_iid{args.iid}_localep{args.local_ep}_lam{args.lam}.pt")
                    best_acc = acc
                del ckpt

        # testing
        local.test(model.to(args.device))
