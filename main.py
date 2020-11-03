import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.sampling import load_data, data_iid, data_noniid
from utils.options import args_parser
from utils.Fed import FedAvg
from model.Update import LocalUpdate
from model import models


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)

    # load dataset and distribute to users
    source_name = "amazon"
    target_name = "webcam"

    print('Src: %s, Tar: %s' % (source_name, target_name))
    source_data, target_train_data, target_test_data = load_data(source_name, target_name, data_dir='/data/xian/Office-31/')

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
        for epoch in range(args.rounds):
            w_locals, loss_locals = [], []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            j=0
            for idx in idxs_users:
                local = LocalUpdate(args, source_data, target_train_data, target_test_data, idxs=dict_users[idx])
                print(j)
                j = j + 1
                w, loss, optimizer = local.train(copy.deepcopy(model).to(args.device))
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
            writer.add_scalar("loss_avg", loss_avg, global_step = epoch)

            # evaluation and save
            if epoch % args.eval_interval == 0:
                # evaluate
                acc = local.test(model.to(args.device))
                writer.add_scalar("accuracy", acc, global_step = epoch)
                # save last, best and delete
                ckpt = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer}
                torch.save(ckpt, f"./save/last_iid{args.iid}.pt")
                if acc > best_acc  and epoch < args.rounds-1:
                    torch.save(ckpt, f"./save/best_iid{args.iid}.pt")
                    best_acc = acc
                del ckpt

        # plot loss curve
        plt.figure()
        plt.plot(range(len(loss_train)), loss_train)
        plt.ylabel('train_loss')
        plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(source_name, target_name, args.rounds, args.frac, args.iid))
