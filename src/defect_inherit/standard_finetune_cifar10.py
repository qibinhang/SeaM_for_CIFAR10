# JUST for Action, dog, MIT

import argparse
import os.path
import sys
import torch
from torch.utils.data import DataLoader
sys.path.append('../')
sys.path.append('../..')
from defect_inherit.models.resnet import resnet18, resnet50
from utils.dataset_loader import load_dataset
from defect_inherit.finetuner_tmp import finetune
from defect_inherit.config import load_config
import torchvision.transforms as transforms
import torchvision
import numpy as np
import copy


def standard_finetune(model, train_loader, test_loader, n_epochs, lr, momentum, weight_decay, save_path):
    # finetune all parameters.
    fc_module = model.fc
    ignored_params = list(map(id, fc_module.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())

    optim = torch.optim.SGD(
        [
            {'params': base_params},
            {'params': fc_module.parameters(), 'lr': lr * 10}
        ],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model_ft, best_acc, best_epoch = finetune(model, optim, train_loader, test_loader, n_epochs=n_epochs,
                                              save_dir=os.path.dirname(save_path))
    model_ft_params = model_ft.state_dict()
    torch.save(model_ft_params, save_path)
    print()
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Acc  : {best_acc:.2%}')
    print(f'Finish Fine-tuning.\n\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet18'], default='resnet18')
    # parser.add_argument('--dataset', type=str,
    #                     choices=['cifar10'], default='cifar10')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    args = parser.parse_args()
    return args



def load_cifar10(dataset_dir, is_train, labels=None, batch_size=64, num_workers=0, pin_memory=False, is_random=True, part_train=-1):
    """airplane	 automobile	 bird	 cat	 deer	 dog	 frog	 horse	 ship	 truck"""
    if labels is None:
        labels = list(range(10))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    if is_train:
        if is_random:
            transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize])

        train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, transform=transform)
        train_targets = np.array(train.targets)
        idx = np.isin(train_targets, labels)
        target_label = train_targets[idx].tolist()
        trans_label = [labels.index(i) for i in target_label]
        train.targets = trans_label
        train.data = train.data[idx]

        idx = list(range(len(train)))
        np.random.seed(1009)
        np.random.shuffle(idx)
        train_idx = idx[: int(0.8 * len(idx))]
        valid_idx = idx[int(0.8 * len(idx)):]

        if part_train > 0:
            train_idx = train_idx[:part_train]

        train_set = copy.deepcopy(train)
        train_set.targets = [train.targets[idx] for idx in train_idx]
        train_set.data = train.data[train_idx]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=is_random,
                                  num_workers=num_workers, pin_memory=pin_memory)

        valid_set = copy.deepcopy(train)
        valid_set.targets = [train.targets[idx] for idx in valid_idx]
        valid_set.data = train.data[valid_idx]
        valid_loader = DataLoader(valid_set, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, valid_loader

    else:
        test = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                            transform=transform)

        test_targets = np.array(test.targets)
        idx = np.isin(test_targets, labels)
        target_label = test_targets[idx].tolist()
        trans_label = [labels.index(i) for i in target_label]
        test.targets = trans_label
        test.data = test.data[idx]

        test_loader = DataLoader(test, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        return test_loader



def main():
    args = get_args()
    print(args)
    configs = load_config()
    num_workers = 8
    pin_memory = True
    save_path = f'{configs.standard_finetune_dir}_cifar10/{args.model}_cifar10_dropout_{args.dropout}/model_ft.pth'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    train_loader, val_loader = load_cifar10(dataset_dir='../../data/dataset', is_train=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = load_cifar10(dataset_dir='../../data/dataset', is_train=False)

    num_classes = 10
    model = eval(args.model)(pretrained=True, dropout=args.dropout, num_classes=num_classes).to('cuda')

    print(f'\n\n## Start Fine-tuning ##\n\n')
    print(model)

    standard_finetune(model, train_loader, test_loader,
                      args.n_epochs, args.lr, args.momentum, args.weight_decay,
                      save_path)


if __name__ == '__main__':
   main()
