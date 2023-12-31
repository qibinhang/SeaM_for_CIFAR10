import argparse
import os.path
import sys
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
sys.path.append('../')
sys.path.append('../../')
from defect_inherit.models.resnet import resnet18, resnet50
from defect_inherit.reengineer import Reengineer
from utils.dataset_loader import load_dataset
from defect_inherit.finetuner import finetune
from defect_inherit.config import load_config

import torchvision.transforms as transforms
import torchvision
import numpy as np
import copy


def step_1(train_loader, test_loader):
    print('Step 1: Only finetune the output layer.')

    if os.path.exists(step_1_path):
        print(f'Loading {step_1_path}...')
        model_ft_params = torch.load(step_1_path, map_location=torch.device('cuda'))
        return model_ft_params
    num_classes = 10
    model_pt = eval(model_name)(pretrained=True, num_classes=num_classes).to('cuda')
    output_layer = model_pt.fc
    output_layer_params_id = list(map(id, output_layer.parameters()))
    hidden_layer_params = filter(lambda p: id(p) not in output_layer_params_id,
                                 model_pt.parameters())

    for p in hidden_layer_params:
        p.requires_grad = False
    optim = torch.optim.SGD(
        output_layer.parameters(),
        lr=lr_output_layer_step_1,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model_ft, best_acc, best_epoch = finetune(model_pt, optim, train_loader, test_loader,
                                              n_epochs=10, early_stop=early_stop)
    model_ft_params = model_ft.state_dict()
    torch.save(model_ft_params, step_1_path)
    print()
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Acc  : {best_acc:.2%}')
    print(f'Step 1 Finished.\n\n')
    return model_ft_params


def step_2(model_ft_params, train_loader, test_loader):
    print(f'Step 2: Modularize the fine-tuned model obtained in Step 1')

    if os.path.exists(step_2_path):
        print(f'Loading {step_2_path}...')
        module_info = torch.load(step_2_path)
        return module_info

    num_classes = 10
    model_modular = eval(model_name)(pretrained=False, num_classes=num_classes, is_reengineering=True).to('cuda')
    model_modular_params = model_modular.state_dict()
    model_modular_params.update(model_ft_params)
    model_modular.load_state_dict(model_modular_params)

    modularizer = Reengineer(model_modular, train_loader, test_loader)
    masks, output_layer = modularizer.alter(lr_mask=lr_mask, lr_output_layer=lr_output_layer_step_2,
                                            n_epochs=30, alpha=alpha, prune_threshold=prune_threshold,
                                            early_stop=early_stop)
    masks.update(output_layer)
    torch.save(masks, step_2_path)
    return masks


def step_3(module_info, train_loader, test_loader):
    print(f'Step 3: Finetune according to modularization results.')

    num_classes = 10
    model_pt = eval(model_name)(pretrained=True, dropout=dropout, num_classes=num_classes).to('cuda')

    # remove irrelevant weights using masks.
    model_pt_params = model_pt.state_dict()
    masked_params = OrderedDict()
    for name, weight in model_pt_params.items():
        if f'{name}_mask' in module_info:
            mask = module_info[f'{name}_mask']
            bin_mask = (mask > 0).float()
            masked_weight = weight * bin_mask
            masked_params[name] = masked_weight
        else:
            masked_params[name] = weight

    # load pretrained output layer
    # masked_params['fc.weight'] = module_info['fc.weight']
    # masked_params['fc.bias'] = module_info['fc.bias']

    model_pt.load_state_dict(masked_params)

    for p in model_pt.parameters():
        p.requires_grad = True

    output_layer = model_pt.fc
    output_layer_params_id = list(map(id, output_layer.parameters()))
    hidden_layer_params = filter(lambda p: id(p) not in output_layer_params_id,
                                 model_pt.parameters())

    optim = torch.optim.SGD(
        [
            {'params': hidden_layer_params},
            {'params': output_layer.parameters(), 'lr': lr_output_layer_step_3}
        ],
        lr=lr_hidden_layer,
        momentum=momentum,
        weight_decay=weight_decay,
    )


    print(f'\n\n## Start Fine-tuning ##\n\n')
    print(model_pt)

    model_ft, best_acc, best_epoch = finetune(model_pt, optim, train_loader, test_loader,
                                              n_epochs=30, early_stop=early_stop)
    model_ft_params = model_ft.state_dict()
    torch.save(model_ft_params, step_3_path)
    print()
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Acc  : {best_acc:.2%}')
    print(f'Step 3 Finished.\n\n')
    return model_ft_params



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet18'], default='resnet18')
    # parser.add_argument('--dataset', type=str,
    #                     choices=['cub200', 'dog120', 'flower102', 'mit67', 'action40'], required=True)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--lr_s1_output_layer', type=float, default=0.1,
                        help='learning rate for optimizing the output layer in step 1.')

    parser.add_argument('--lr_s2_output_layer', type=float, default=0.001,
                        help='learning rate for optimizing the output layer in step 2.')
    parser.add_argument('--lr_s2_mask', type=float, default=0.1,
                        help='learning rate for optimizing the mask in step 2.')
    parser.add_argument('--alpha', type=float, default=1,
                        help='the weight for the weighted sum of two losses in modularization.')
    parser.add_argument('--prune_threshold', type=float, default=1.0)

    parser.add_argument('--lr_s3_output_layer', type=float, default=0.05,
                        help='learning rate for optimizing the output layer in step 3.')
    parser.add_argument('--lr_s3_hidden_layer', type=float, default=0.005,
                        help='learning rate for optimizing the hidden layer in step 3.')

    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--early_stop', type=int, default=30)
    args = parser.parse_args()
    return args


def modular_finetune():
    train_loader, val_loader = load_cifar10(dataset_dir='../../data/dataset', is_train=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = load_cifar10(dataset_dir='../../data/dataset', is_train=False)

    print(f'\n====== Start Step 1 ======\n')
    model_ft_params = step_1(train_loader, test_loader)

    print(f'\n====== Start Step 2 ======')
    module_info = step_2(model_ft_params, train_loader, test_loader)

    print(f'\n====== Start Step 3 ======\n')
    step_3(module_info, train_loader, test_loader)




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


if __name__ == '__main__':
    args = get_args()
    print(args)

    num_workers = 8
    pin_memory = True

    model_name = args.model
    dataset_name = 'cifar10'

    dropout = args.dropout
    # num_classes = 200

    # lr for step 1
    lr_output_layer_step_1 = args.lr_s1_output_layer

    # lr for step 2
    lr_output_layer_step_2 = args.lr_s2_output_layer
    lr_mask = args.lr_s2_mask
    alpha = args.alpha
    prune_threshold = args.prune_threshold
    assert 0 <= prune_threshold <= 1.0

    # lr for step 3
    lr_output_layer_step_3 = args.lr_s3_output_layer
    lr_hidden_layer = args.lr_s3_hidden_layer

    momentum = args.momentum
    weight_decay = args.weight_decay

    n_epochs = args.n_epochs
    early_stop = args.early_stop

    configs = load_config()

    save_dir = f'{configs.seam_finetune_dir}_cifar10/{args.model}_cifar10_dropout_{args.dropout}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    step_1_path = f'{save_dir}/step_1_model_ft.pth'
    step_2_path = f'{save_dir}/lr_mask_{lr_mask}_alpha_{alpha}_thres_{prune_threshold}/step_2_module.pth'
    step_3_path = f'{save_dir}/lr_mask_{lr_mask}_alpha_{alpha}_thres_{prune_threshold}/step_3_module_ft.pth'

    if not os.path.exists(os.path.dirname(step_2_path)):
        os.makedirs(os.path.dirname(step_2_path))

    print(f'step_1: {step_1_path}')
    print(f'step_2: {step_2_path}')
    print(f'step_3: {step_3_path}')


    modular_finetune()
