import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help="rounds of training")
    parser.add_argument('--num_users', type=int,
                        default=15, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2,
                        help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--bs', type=int, default=16, help="test batch size")

    # model arguments
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="SGD momentum (default: 0.9)")
    parser.add_argument('--n_class', type=int, default=31,
                        help="Number of classes")
    parser.add_argument('--lam', type=float, default=10,
                        help="lambda of transfer loss")
    parser.add_argument('--l2_decay', type=float, default=0, help="l2_deacy")

    # other arguments
    parser.add_argument('--iid', action='store_true',
                        help='whether iid or not')
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--eval_interval', type=int, default=1)
    args = parser.parse_args()
    return args
