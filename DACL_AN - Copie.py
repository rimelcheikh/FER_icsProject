import argparse
import random
import pprint
import time
import sys
import os

from datetime import timedelta
from workspace import Workspace

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models_DACL import resnet18
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils_DACL import Logger, AverageMeter, accuracy, calc_metrics, RandomFiveCrop
from utils import load_data
from tqdm import tqdm

# centerloss module
from loss import SparseCenterLoss

parser = argparse.ArgumentParser(description='DACL for FER in the wild')
parser.add_argument('--arch', type=str, default="resnet18")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float)
parser.add_argument('--bs', type=int, default=25)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--lamb', type=float, default=0.01)
parser.add_argument('--pretrained', type=str, default='msceleb')
parser.add_argument('--deterministic', default=False, action='store_true')


def main(cfg):

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if cfg['deterministic']:
        random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        cudnn.deterministic = True
        cudnn.benchmark = False

    # Loading RAF-DB
    # -----------------
    print('[>] Loading dataset '.ljust(64, '-'))

    # train set

    data_transforms = [transforms.ColorJitter(brightness=0.5, contrast=0.5),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomAffine(degrees=30,
                                                translate=(.1, .1),
                                                scale=(1.0, 1.25),
                                                interpolation=transforms.InterpolationMode.BILINEAR)]
    
    train_data = load_data.AffectNetCategorical(idx_set=0,
                                   max_loaded_images_per_label=5000,
                                   transforms=transforms.Compose(data_transforms),
                                   base_path_to_affectnet='C:/Users/rielcheikh/Desktop/FER/DB/AffectNet/')
    train_loader = DataLoader(train_data, 
                                      batch_size=32, 
                                      shuffle=True, 			
                                      num_workers=2,
                                      pin_memory = True)

    # validation set
    val_data = load_data.AffectNetCategorical(idx_set=1,
                             max_loaded_images_per_label=100000,
                             transforms=None,
                             base_path_to_affectnet='C:/Users/rielcheikh/Desktop/FER/DB/AffectNet/')
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory = True)
    
    print('[*] Loaded dataset!')

    # Create Model
    # ------------
    print('[>] Model '.ljust(64, '-'))
    if cfg['arch'] == 'resnet18':
        feat_size = 512
        if not cfg['pretrained'] == '':
            model = resnet18(pretrained=cfg['pretrained'])
            model.fc = nn.Linear(feat_size, 8)
        else:
            print('[!] model is trained from scratch!')
            model = resnet18(num_classes=8, pretrained=cfg['pretrained'])
    else:
        raise NotImplementedError('only working with "resnet18" now! check cfg["arch"]')
    model = torch.nn.DataParallel(model).to(device)
    print('[*] Model initialized!')

    # define loss function (criterion) and optimizer
    # ----------------------------------------------
    criterion = {
        'softmax': nn.CrossEntropyLoss().to(device),
        'center': SparseCenterLoss(8, feat_size).to(device)
    }
    optimizer = {
        'softmax': torch.optim.Adam(params= model.parameters(), weight_decay= 1e-4),
        'center': torch.optim.Adam(params= model.parameters(), weight_decay= 1e-4),
    }
    """'softmax': torch.optim.SGD(model.parameters(), cfg['lr'],
                                   momentum=cfg['momentum'],
                                   weight_decay=cfg['weight_decay']),
        'center': torch.optim.SGD(criterion['center'].parameters(), cfg['alpha'])"""
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer['softmax'], step_size=10, gamma=0.5, last_epoch=-1)

    # training and evaluation
    # -----------------------
    global best_valid
    best_valid = dict.fromkeys(['acc', 'rec', 'f1', 'aucpr', 'aucroc'], 0.0)

    print('[>] Begin Training '.ljust(64, '-'))
    for epoch in range(1, cfg['epochs'] + 1):

        start = time.time()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, cfg)
        # validate for one epoch
        validate(val_loader, model, criterion, epoch, cfg)

        # progress
        end = time.time()
        progress = (
            f'[*] epoch time = {end - start:.2f}s | '
            f'lr = {optimizer["softmax"].param_groups[0]["lr"]}\n'
        )
        print(progress)

        # lr step
        scheduler.step()

    # best valid info
    # ---------------
    print('[>] Best Valid '.ljust(64, '-'))
    stat = (
        f'[+] acc={best_valid["acc"]:.4f}\n'
        f'[+] rec={best_valid["rec"]:.4f}\n'
        f'[+] f1={best_valid["f1"]:.4f}\n'
        f'[+] aucpr={best_valid["aucpr"]:.4f}\n'
        f'[+] aucroc={best_valid["aucroc"]:.4f}'
    )
    print(stat)


def train(train_loader, model, criterion, optimizer, epoch, cfg):
    losses = {
        'softmax': AverageMeter(),
        'center': AverageMeter(),
        'total': AverageMeter()
    }
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []

    # switch to train mode
    model.train()

    with tqdm(total=int(len(train_loader.dataset) / cfg['batch_size'])) as pbar:
        for i, (images, target, _) in enumerate(train_loader):
            images = images.to(device)
            #target = target.clone().detach()
            target = torch.tensor(target, dtype=torch.long)
            target = target.to(device)

            # compute output
            feat, output, A = model(images)
            l_softmax = criterion['softmax'](output, target)
            l_center = criterion['center'](feat, A, target)
            l_total = l_softmax + cfg['lamb'] * l_center

            # measure accuracy and record loss
            acc, pred = accuracy(output, target)
            losses['softmax'].update(l_softmax.item(), images.size(0))
            losses['center'].update(l_center.item(), images.size(0))
            losses['total'].update(l_total.item(), images.size(0))
            accs.update(acc.item(), images.size(0))

            # collect for metrics
            y_pred.append(pred)
            y_true.append(target)
            y_scores.append(output.data)

            # compute grads + opt step
            optimizer['softmax'].zero_grad()
            optimizer['center'].zero_grad()
            l_total.backward()
            optimizer['softmax'].step()
            optimizer['center'].step()

            # progressbar
            pbar.set_description(f'TRAINING [{epoch:03d}/{cfg["epochs"]}]')
            pbar.set_postfix({'L': losses["total"].avg,
                              'Ls': losses["softmax"].avg,
                              'Lsc': losses["center"].avg,
                              'acc': accs.avg})
            pbar.update(1)

    metrics = calc_metrics(y_pred, y_true, y_scores)
    progress = (
        f'[-] TRAIN [{epoch:03d}/{cfg["epochs"]}] | '
        f'L={losses["total"].avg:.4f} | '
        f'Ls={losses["softmax"].avg:.4f} | '
        f'Lsc={losses["center"].avg:.4f} | '
        f'acc={accs.avg:.4f} | '
        f'rec={metrics["rec"]:.4f} | '
        f'f1={metrics["f1"]:.4f} | '
        f'aucpr={metrics["aucpr"]:.4f} | '
        f'aucroc={metrics["aucroc"]:.4f}'
    )
    print(progress)
    write_log(losses, accs.avg, metrics, epoch, tag='train')


def validate(valid_loader, model, criterion, epoch, cfg):
    losses = {
        'softmax': AverageMeter(),
        'center': AverageMeter(),
        'total': AverageMeter()
    }
    accs = AverageMeter()
    y_pred, y_true, y_scores = [], [], []

    # switch to evaluate mode
    model.eval()

    with tqdm(total=int(len(valid_loader.dataset) / cfg['batch_size'])) as pbar:
        with torch.no_grad():
            for i, (images, target, _) in enumerate(valid_loader):

                images = images.to(device)
                #target = target.clone().detach()
                target = torch.tensor(target, dtype=torch.long)
                target = target.to(device)

                # compute output
                feat, output, A = model(images)
                l_softmax = criterion['softmax'](output, target)
                l_center = criterion['center'](feat, A, target)
                l_total = l_softmax + cfg['lamb'] * l_center

                # measure accuracy and record loss
                acc, pred = accuracy(output, target)
                losses['softmax'].update(l_softmax.item(), images.size(0))
                losses['center'].update(l_center.item(), images.size(0))
                losses['total'].update(l_total.item(), images.size(0))
                accs.update(acc.item(), images.size(0))

                # collect for metrics
                y_pred.append(pred)
                y_true.append(target)
                y_scores.append(output.data)

                # progressbar
                pbar.set_description(f'VALIDATING [{epoch:03d}/{cfg["epochs"]}]')
                pbar.update(1)

    metrics = calc_metrics(y_pred, y_true, y_scores)
    progress = (
        f'[-] VALID [{epoch:03d}/{cfg["epochs"]}] | '
        f'L={losses["total"].avg:.4f} | '
        f'Ls={losses["softmax"].avg:.4f} | '
        f'Lsc={losses["center"].avg:.4f} | '
        f'acc={accs.avg:.4f} | '
        f'rec={metrics["rec"]:.4f} | '
        f'f1={metrics["f1"]:.4f} | '
        f'aucpr={metrics["aucpr"]:.4f} | '
        f'aucroc={metrics["aucroc"]:.4f}'
    )
    print(progress)

    # save model checkpoints for best valid
    if accs.avg > best_valid['acc']:
        save_checkpoint(epoch, model, cfg, tag='best_valid_acc.pth')
    if metrics['rec'] > best_valid['rec']:
        save_checkpoint(epoch, model, cfg, tag='best_valid_rec.pth')

    best_valid['acc'] = max(best_valid['acc'], accs.avg)
    best_valid['rec'] = max(best_valid['rec'], metrics['rec'])
    best_valid['f1'] = max(best_valid['f1'], metrics['f1'])
    best_valid['aucpr'] = max(best_valid['aucpr'], metrics['aucpr'])
    best_valid['aucroc'] = max(best_valid['aucroc'], metrics['aucroc'])
    write_log(losses, accs.avg, metrics, epoch, tag='valid')


def write_log(losses, acc, metrics, e, tag='set'):
    # tensorboard
    writer.add_scalar(f'L_softmax/{tag}', losses['softmax'].avg, e)
    writer.add_scalar(f'L_center/{tag}', losses['center'].avg, e)
    writer.add_scalar(f'L_total/{tag}', losses['total'].avg, e)
    writer.add_scalar(f'acc/{tag}', acc, e)
    writer.add_scalar(f'rec/{tag}', metrics['rec'], e)
    writer.add_scalar(f'f1/{tag}', metrics['f1'], e)
    writer.add_scalar(f'aucpr/{tag}', metrics['aucpr'], e)
    writer.add_scalar(f'aucroc/{tag}', metrics['aucroc'], e)


def save_checkpoint(epoch, model, cfg, tag):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(cfg['save_path'], tag))


if __name__ == '__main__':

    # setting up workspace
    args = parser.parse_args()
    workspace = Workspace(args)
    cfg = workspace.config

    # setting up writers
    global writer
    writer = SummaryWriter(cfg['save_path'])
    sys.stdout = Logger(os.path.join(cfg['save_path'], 'log.log'))

    # print finalized parameter config
    print('[>] Configuration '.ljust(64, '-'))
    pp = pprint.PrettyPrinter(indent=2)
    print(pp.pformat(cfg))

    # -----------------
    start = time.time()
    main(cfg)
    end = time.time()
    # -----------------

    print('\n[*] Fini! '.ljust(64, '-'))
    print(f'[!] total time = {timedelta(seconds=end - start)}s')
    sys.stdout.flush()