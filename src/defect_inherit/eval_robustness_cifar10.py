import argparse
import torch
import sys
sys.path.append('..')
sys.path.append('../..')
from torch.utils.data import DataLoader
from tqdm import tqdm
from advertorch.attacks import LinfPGDAttack
from utils.dataset_loader import load_dataset
from defect_inherit.models.resnet import resnet18, resnet18_nofc, resnet50, resnet50_nofc
from defect_inherit.config import load_config
import torchvision.transforms as transforms
import torchvision
import numpy as np
import copy


"""Copy from ICSE'22 ReMos. Thanks the authors!"""

def adversary_test(model, loader, adversary):
    model.eval()
    total = 0
    top1_clean = 0
    top1_adv = 0
    adv_success = 0
    adv_trial = 0
    for batch, label in tqdm(loader, ncols=80, desc='adv_test'):
        batch, label = batch.to('cuda'), label.to('cuda')

        total += batch.size(0)
        out_clean = model(batch)
        y = torch.zeros(batch.shape[0], model.fc.in_features).cuda()  # for ResNet
        y[:, 0] = 1000
        batch_adv = adversary.perturb(batch, y)

        out_adv = model(batch_adv)

        pred_clean = torch.argmax(out_clean, dim=1)
        pred_adv = torch.argmax(out_adv, dim=1)

        clean_correct = pred_clean.eq(label)
        adv_trial += int(clean_correct.sum().item())
        adv_success += int(pred_adv[clean_correct].eq(label[clean_correct]).sum().detach().item())
        top1_clean += int(pred_clean.eq(label).sum().detach().item())
        top1_adv += int(pred_adv.eq(label).sum().detach().item())

    return float(top1_clean) / total * 100, float(top1_adv) / total * 100, float(
        adv_trial - adv_success) / adv_trial * 100


def myloss(yhat, y):
    return -((yhat[:, 0] - y[:, 0]) ** 2 + 0.1 * ((yhat[:, 1:] - y[:, 1:]) ** 2).mean(1)).mean()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet18'], default='resnet18')
    # parser.add_argument('--dataset', type=str,
    #                     choices=['cub200', 'dog120', 'flower102', 'mit67', 'action40'], required=True)

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--eval_method', type=str, choices=['seam', 'remos', 'standard', 'retrain'], required=True)

    # just for seam_finetune
    parser.add_argument('--lr_mask', type=float, default=0.0,
                        help='learning rate for optimizing the mask in step 2.')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='the weight for the weighted sum of two losses in re-engineering.')
    parser.add_argument('--prune_threshold', type=float, default=1.0)
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




if __name__ == '__main__':
    args = get_args()
    configs = load_config()
    num_workers = 8
    pin_memory = True

    eval_method = args.eval_method

    if eval_method == 'seam':
        model_path = f'{configs.seam_finetune_dir}_cifar10/{args.model}_cifar10_dropout_{args.dropout}/' \
                     f'lr_mask_{args.lr_mask}_alpha_{args.alpha}_thres_{args.prune_threshold}/step_3_module_ft.pth'
    elif eval_method == 'standard':
        model_path = f'{configs.standard_finetune_dir}_cifar10/{args.model}_cifar10_dropout_{args.dropout}/model_ft.pth'
    elif eval_method == 'retrain':
        model_path = f'{configs.retrain_dir}/{args.model}_cifar10_dropout_{args.dropout}/model_rt.pth'
    else:
        raise ValueError

    test_loader = load_cifar10(dataset_dir='../../data/dataset', is_train=False)
    num_classes = 10

    model_ft = eval(args.model)(pretrained=False, num_classes=num_classes).to('cuda').eval()
    model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))

    model_pt = eval(f'{args.model}_nofc')(pretrained=True, num_classes=num_classes).to('cuda').eval()

    attacker = LinfPGDAttack(
        model_pt, loss_fn=myloss, eps=0.1,
        nb_iter=100, eps_iter=0.01,
        rand_init=True, clip_min=-2.2, clip_max=2.2,
        targeted=False)

    clean_top1, _, adv_sr = adversary_test(model_ft, test_loader, attacker)

    print('Clean Top-1: {:.2f} | Attack Success Rate: {:.2f}'.format(clean_top1, adv_sr))
