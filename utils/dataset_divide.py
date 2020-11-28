import numpy as np

def iid_sampling(args, dict_idxs, dict_classes, dataset, iid_classes):
    """
    For iid_classes, do sampling as evenly as possible
    """
    assigned_sample = 0

    for i, class_idx in enumerate(iid_classes):     # assign samples of each class to client

        # print(dataset[dataset["label"]==class_idx]["idx"].values[:, 0])

        sample_idxs = dataset[dataset["label"]==class_idx].index.values
        for sample_idx in sample_idxs:
            client_idx = assigned_sample % args.num_users
            dict_idxs[client_idx].append(sample_idx)
            dict_classes[client_idx].append(class_idx)           
            assigned_sample += 1

    return dict_idxs, dict_classes 


def non_iid_sampling(args, dict_idxs, dict_classes, dataset, non_iid_classes):
    """
    For non_iid_classes, do sampling as unevenly as possible
    """
    assigned_sample = 0
    frame_per_client = args.num_sample*len(non_iid_classes)/args.num_users

    for i, class_idx in enumerate(non_iid_classes):     # assign samples of each class to client
        sample_idxs = dataset[dataset["label"]==class_idx].index.values
        for sample_idx in sample_idxs:
            client_idx = assigned_sample // frame_per_client
            dict_idxs[client_idx].append(sample_idx)
            dict_classes[client_idx].append(class_idx)           
            assigned_sample += 1

    return dict_idxs, dict_classes 


def random_sampling(args, dataset):
    """
    Do sampling randomly using random seed on the whole dataset
    """
    dict_idxs = {i: [] for i in range(args.num_users)}
    dict_classes = {i: [] for i in range(args.num_users)}

    total_data_num = len(dataset)
    per_client_data_num = total_data_num // args.num_users
    all_idxs = [i for i in range(total_data_num)]

    np.random.seed(args.seed)
    for i in range(args.num_users):
        dict_idxs[i] = set(np.random.choice(all_idxs, per_client_data_num, replace=False))
        all_idxs = list(set(all_idxs) - dict_idxs[i])
        dict_classes[i] = dataset["label"].loc[dict_idxs[i]].values

    return dict_idxs, dict_classes


def data_sampling(dataset, args):
    """
    Sample data for per user according to non-iidness 
    ---
    Args:
    - dataset: a class of torch.utils.data.Dataset
    - args
    Return:
    - dict_idxs: a dictionary. Keys: client idx; values: idxs in dataset (the first argument) for each client 
    - dict_classes: a dictionary. Keys: client idx; values: class idxs for each client 
    """
    dict_idxs = {i: [] for i in range(args.num_users)}
    dict_classes = {i: [] for i in range(args.num_users)}

    dataset = dataset.imgs

    # control the non-iidness
    non_iid_classes = set(np.random.choice([*range(100,200)], int(100*args.non_iidness), replace=False))
    iid_classes = set([*range(100,200)]) - non_iid_classes
    dict_idxs, dict_classes = iid_sampling(args, dict_idxs, dict_classes, dataset, iid_classes)
    dict_idxs, dict_classes = non_iid_sampling(args, dict_idxs, dict_classes, dataset, non_iid_classes)

    # or use a random seed 
    # dict_idxs, dict_classes = random_sampling(args, dict_idxs, dict_classes, dataset)

    return dict_idxs, dict_classes