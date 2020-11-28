import os
from functools import wraps

import numpy as np
import torch
from torchvision import datasets


def calulate_non_iidness(args, dict_classes):
    """
    Calulate the non-iidness in terms of the class distribution.
    """
    K = args.num_users      # num of clients
    num_classes = 31
    m = len(dict_classes[0])    # num of samples on per client

    array = np.eye(num_classes)
    class_stat = []    # shape: (#clients, #classes)
    for (key, value) in dict_classes.items():
        class_stat.append(array[value].sum(0))
    class_stat = np.array(class_stat)

    sum = 0
    for i, row_i in enumerate(class_stat):
        for row_j in class_stat[i:]:
            sum += np.linalg.norm(row_i-row_j, ord=1)
    non_iidness = sum/(K*(K-1)*m)

    return non_iidness


def data_iid(dataset, num_users):

    dict_idxs = dict(zip([*range(num_users)], [[] for i in range(num_users)]))
    dict_classes = dict(zip([*range(num_users)], [[] for i in range(num_users)]))
    all_idxs = np.array([i for i in range(len(dataset))])
    
    class_list = []
    print("Going through the dataset...")
    for _, label in iter(dataset):
        class_list.append(label)
    class_list = np.array(class_list)   # a list of the labels of a datset

    assigned_sample = 0
    iid_classes = len(set(class_list))

    # assign samples of each class
    for class_idx in range(iid_classes):     

        sample_idxs = all_idxs[class_list==class_idx]

        # assign samples belonging to the same class 
        for sample_idx in sample_idxs:
            client_idx = assigned_sample % num_users
            dict_idxs[client_idx].append(sample_idx)
            dict_classes[client_idx].append(class_idx)           
            assigned_sample += 1

    return dict_idxs, dict_classes 


def data_noniid(dataset, num_users):
    dict_idxs = {i: np.array([], dtype='int64') for i in range(num_users)}
    num_imgs = len(dataset) // num_users
    idxs = np.arange(len(dataset))
    rand = range(num_users)

    class_list, dict_classes = [], {}
    print("Going through the dataset...")
    for _, label in iter(dataset):
        class_list.append(label)
    class_list = np.array(class_list)   # a list of the labels of a datset

    for i in range(num_users):
        rand_set = np.random.choice(rand, 1, replace=False)
        rand = list(set(rand) - set(rand_set))
        dict_idxs[i] = idxs[int(rand_set) *
                             num_imgs:(int(rand_set) + 1) * num_imgs]
        dict_classes[i] = class_list[list(dict_idxs[i])]

    return dict_idxs, dict_classes


def load_data(src, tar, data_dir='dataset', use_cv2=False):
    folder_src = os.path.join(os.path.join(data_dir, src), 'images')
    folder_tar = os.path.join(os.path.join(data_dir, tar), 'images')

    if use_cv2:
        import cv2
        from opencv_transforms import transforms

        def loader_opencv(path: str) -> np.ndarray:
            return cv2.imread(path)

        transform = {
            'train': transforms.Compose(
                [transforms.Resize((256, 256), interpolation=cv2.INTER_LINEAR),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]),
            'test': transforms.Compose(
                [transforms.Resize((224, 224), interpolation=cv2.INTER_LINEAR),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
        }

        source_data = datasets.ImageFolder(
            root=folder_src, transform=transform['train'], loader=loader_opencv)
        # source_data_loader = torch.utils.data.DataLoader(source_data, batch_size=batch_size, shuffle=True, **kwargs, drop_last = True)

        target_train_data = datasets.ImageFolder(
            root=folder_tar, transform=transform['train'], loader=loader_opencv)
        # target_train_loader = torch.utils.data.DataLoader(target_train_data, batch_size=batch_size, shuffle=True, **kwargs, drop_last = True)

        target_test_data = datasets.ImageFolder(
            root=folder_tar, transform=transform['test'], loader=loader_opencv)
        # target_test_loader = torch.utils.data.DataLoader(target_test_data, batch_size=batch_size, shuffle=True, **kwargs, drop_last = False)
    else:
        from torchvision import transforms
        transform = {
            'train': transforms.Compose(
                [transforms.Resize((256, 256)),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]),
            'test': transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
        }

        source_data = datasets.ImageFolder(
            root=folder_src, transform=transform['train'])
        # source_data_loader = torch.utils.data.DataLoader(source_data, batch_size=batch_size, shuffle=True, **kwargs, drop_last = True)

        target_train_data = datasets.ImageFolder(
            root=folder_tar, transform=transform['train'])
        # target_train_loader = torch.utils.data.DataLoader(target_train_data, batch_size=batch_size, shuffle=True, **kwargs, drop_last = True)

        target_test_data = datasets.ImageFolder(
            root=folder_tar, transform=transform['test'])
        # target_test_loader = torch.utils.data.DataLoader(target_test_data, batch_size=batch_size, shuffle=True, **kwargs, drop_last = False)

    return source_data, target_train_data, target_test_data

def load_unlabeled_data(tar, data_dir='dataset/'):
    from torchvision import transforms

    folder_tar = data_dir + tar + '/images'

    transform = {
        'weak': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]),
        'strong': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    }

    unlabeled_weak_data = datasets.ImageFolder(root=folder_tar, transform=transform['weak'])
    unlabeled_strong_data = datasets.ImageFolder(root=folder_tar, transform=transform['strong'])

    return unlabeled_weak_data, unlabeled_strong_data