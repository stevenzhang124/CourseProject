import os
from functools import wraps

import numpy as np

import torch
from torchvision import datasets


def data_iid(dataset, num_users):
    num_item = len(dataset) // num_users
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(
            all_idxs, num_item, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def data_noniid(dataset, num_users):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    num_imgs = len(dataset) // num_users
    idxs = np.arange(len(dataset))
    rand = range(num_users)
    for i in range(num_users):
        rand_set = np.random.choice(rand, 1, replace=False)
        rand = list(set(rand) - set(rand_set))
        dict_users[i] = idxs[int(rand_set) *
                             num_imgs:(int(rand_set) + 1) * num_imgs]

    return dict_users


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
