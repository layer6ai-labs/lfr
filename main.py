import argparse
import copy
import os
import random
from matplotlib import pyplot as plt
import pandas as pd
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from linear_eval import eval, eval_ft
from utils.load_data import load_data
from utils.trainers import train_loop, train_loop_supervised, validation_loop_supervised
from utils.utils import adjust_learning_rate, get_criterion, load_ckpt, load_model, init_optimizer, get_path, get_logger


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='kvasir',  help='path to dataset or torch dataset name')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=300, type=int, metavar='E',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='number of total epochs during warmup')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_with_steps', action='store_true', 
                    help='whether to train with single step per epoch')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='B',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial (base) learning rate, default 0.01', dest='lr')
parser.add_argument('--warmup_lr', '--warmup-learning-rate', default=0, type=float,
                    help='initial warmup learning rate, default 0')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('--eval_wd', '--eval_weight-decay', default=0, type=float,
                    help='weight decay (default: 0) for linear probing')
parser.add_argument('-p', '--print-freq', default=10, type=int, 
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--ckpt_path', type=str, default='ckpt',  help='path to save checkpoints')
parser.add_argument('--eval_epochs', type=int, default=100,  help='linear eval epochs')
parser.add_argument('--eval_lr', '--eval-learning-rate', default=5e-3, type=float)
parser.add_argument('--eval_bs', '--eval-batch-size', default=512, type=int)
parser.add_argument('--early_stop', default=10000, type=int)

parser.add_argument('--random_init_eval', action='store_true',
                    help='calculate linear probing performance with random initialized encoder')
# more specific configs:
parser.add_argument('--method', type=str, default='lfr',  help='training method', 
                    choices=['lfr', 'supervised', 'supervised-aug', 
                             'autoencoder', 'simsiam', 
                             'simclr', 'diet', 'diet-aug', 'stab'])
parser.add_argument('--stab-drop-rate', type=float, default=0.1)
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--num_targets', default=5, type=int,
                    help='# of target networks (default: 5')
parser.add_argument('--pred_dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
# argument for simsiam
parser.add_argument('--proj_dim', default=512, type=int,
                    help='hidden dimension of the projector (default: 512)')
parser.add_argument('--proj_layers', default=3, type=int)

# SimSiam -- fixing predictor learning rate is helpful
parser.add_argument('--fix_pred_lr', action='store_true',
                    help='Fix learning rate for the predictor')
parser.add_argument('--fix_eval_lr', action='store_true',
                    help='Fix evaluation learning rate for the classifier')
parser.add_argument('--fix_lr', action='store_true',
                    help='Fix self-supervised learning rate for the encoder')

# BYOL's 3rd arxiv update, increasing predictor lr by 10X is helpful
parser.add_argument('--pred_lr_scale', default=1, type=int,
                    help='learning rate scale factor of predictor (default: 1')

parser.add_argument('--pred_epochs', default=5, type=int,
                    help='learning rate scale factor of predictor (default: 5')

parser.add_argument('--rerun_training', action='store_true')
parser.add_argument('--rerun_eval', action='store_true')

parser.add_argument('--run_train', default=1, type=int)
parser.add_argument('--run_eval', default=1, type=int)

parser.add_argument('--save_freq', default=1000, type=int)
parser.add_argument('--eval_freq', default=200, type=int)

parser.add_argument('--train-predictor-individually', action='store_true') 

parser.add_argument('--target_layers', nargs='*', type=int, required=True,
                    help='num of layers in the target networks')
parser.add_argument('--random-dropout', action='store_true')   

parser.add_argument('--init-beta', action='store_true')
parser.add_argument('--regularize-weight', action='store_true')

parser.add_argument('--no-bias', action='store_true')

parser.add_argument('--loss', default='cosine', type=str, 
                    choices=['cosine','softmax','barlow-batch', 'mse', 'ce', 'ce-smooth'],
                    help='loss function')

parser.add_argument('--temp', default=0.07, type=float, 
                    help='temperature of loss')

parser.add_argument('--pred_layers', default=2, type=int)

parser.add_argument('--num_of_classes', default=10, type=int)

parser.add_argument('--lambd', default=0.01, type=float)

parser.add_argument('--eval_dataset', default=None, type=str)

parser.add_argument('--num-of-runs', default=1, type=int)

parser.add_argument('--run_id', default=0, type=int)

parser.add_argument('--optimizer-type', default='sgd', type=str)

parser.add_argument('--target_sample_ratio', default=1, type=int,
                    help='how many times of target networks to sample from. \
                    For example, when set to 10, create 10x num_of_target \
                    number of targets and sample from them, default is 1')

parser.add_argument('--finetune_epochs', default=0, type=int)

parser.add_argument('--finetune_lr', default=5e-3, type=float)

parser.add_argument('--train_sample_frac', default=1.0, type=float)


def main():
    args = parser.parse_args()
    if len(args.target_layers) == 1:
        args.target_layers = [args.target_layers[0] for _ in range(args.num_targets)]
    
    if args.method != 'lfr': args.train_predictor_individually = False
    
    # add metrics for datasets
    if args.dataset == 'mimic3-los':
        args.metrics = ['acc', 'kappa']
    else:
        args.metrics = ['acc']

    if args.eval_dataset is None: 
        args.eval_dataset = args.dataset
    if args.num_targets != len(args.target_layers): 
        raise ValueError('Number of target networks {} \
                         does not match layer inputs {}'.\
                            format(args.num_targets, len(args.target_layers)))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_of_runs > 1:
        for r in range(args.num_of_runs):
            args.run_id = r
            print(f"=> running # {args.run_id} ")
            main_worker(args, device)
    else:
        print(f"=> running # {args.run_id} ")
        main_worker(args, device)

def main_worker(args, device):
    logger = get_logger(get_path(args, 'logs/log.txt'))
    if args.run_train:
        model=train(args, logger, device)
  
    if args.run_eval:
        save_res = os.path.join(get_path(args, ''), 'linear_eval/result.csv')
        if args.rerun_eval or not os.path.exists(save_res): 
            last_epoch = args.early_stop - 1 if args.early_stop <= args.epochs else args.epochs - 1
            filename = get_path(args, 'checkpoint_{:04d}.pth.tar'.format(last_epoch))
            train_loader_labelled, test_loader_labelled = load_data(args, args.eval_dataset, labelled=True)
            save_res = os.path.join(get_path(args, ''), 'linear_eval')
            train_res_final, test_res_final, knn_acc_final = eval(args, device, train_loader_labelled, test_loader_labelled, 
                                                                save_rep=None, save_res=save_res, 
                                                                save_name='result', logger=logger, 
                                                                ckpt_path=filename)
            if args.method == 'lfr' and args.random_init_eval:
                # evaluate random intialization
                print("=> Running evaluation for the random init model")
                filename = get_path(args, 'checkpoint_{:04d}.pth.tar'.format(-1))
                train_res_init, test_res_init, knn_acc_init = eval(args, device, train_loader_labelled, test_loader_labelled, 
                                                                save_rep=None, save_res=save_res, 
                                                                save_name='result_0', logger=logger, 
                                                                ckpt_path=filename)
                pd.DataFrame.from_dict({'random init': [knn_acc_init], f'{args.method}': [knn_acc_final]}).to_csv(f'{save_res}/knn.csv', index=False)

    if args.finetune_epochs > 0:
        save_res = os.path.join(get_path(args, ''), f'finetune_eval/train_frac_{args.train_sample_frac}_lr_{args.finetune_lr}/result.csv')
        if args.rerun_eval or not os.path.exists(save_res): 
            last_epoch = args.early_stop - 1 if args.early_stop <= args.epochs else args.epochs - 1
            filename = get_path(args, 'checkpoint_{:04d}.pth.tar'.format(last_epoch))
            train_loader_labelled, test_loader_labelled = load_data(args, args.eval_dataset, labelled=True, train_sample_frac=args.train_sample_frac)
            save_res = os.path.join(get_path(args, ''), f'finetune_eval/train_frac_{args.train_sample_frac}_lr_{args.finetune_lr}')
            train_res_final, test_res_final = eval_ft(args, device, train_loader_labelled, test_loader_labelled, filename,
                                                      save_res=save_res, save_name='result', logger=logger)
            if args.method == 'lfr':
                # evaluate random intialization
                print("=> Running evaluation for the random init model")
                filename = get_path(args, 'checkpoint_{:04d}.pth.tar'.format(-1))
                train_res_init, test_res_init = eval_ft(args, device, train_loader_labelled, test_loader_labelled, filename,
                                                                      save_res=save_res, save_name='result_0', logger=logger)
    for h in logger.handlers: logger.removeHandler(h)

def train(args, logger, device):
    logger.info(args)

    # load data
    logger.info("=> loading data '{}'".format(args.dataset))
    cudnn.benchmark = True
    if 'supervised' in args.method:
        train_loader, test_loader = load_data(args, args.dataset, labelled=True)
    else:
        train_loader, test_loader = load_data(args, args.dataset, labelled=False)

    # create model
    logger.info("=> creating model '{}'".format(args.arch))

    sample_data = None
    if args.target_sample_ratio > 1 and args.method == 'lfr':
        sample_data=next(iter(train_loader))[0].to(device, dtype=torch.float)
    model = load_model(args, device, sample_data=sample_data, num_data=len(train_loader.dataset))

    checkpoint = {'state_dict': model.state_dict()}
    torch.save(checkpoint, get_path(args, 'checkpoint_-001.pth.tar'))

    # define loss function (criterion) and optimizer
    criterion = get_criterion(args, device)
    optimizer, init_lr = init_optimizer(model, args)

    # optionally resume from a checkpoint
    if args.resume:
        load_ckpt(args, optimizer, model, logger)

    logger.info("=> begin training")
    if 'supervised' in args.method: #supervised or supervised-aug
        args.run_eval = 0
        train_supervised(train_loader, model, criterion, optimizer, args, init_lr, device, logger, test_loader)
    else:
        train_ssl(train_loader, model, criterion, optimizer, args, init_lr, device, logger, test_loader)
    logger.info("=> finished training")
    return model


def train_ssl(train_loader_ssl, model, criterion, optimizer, args, init_lr, device, logger, test_loader_ssl=None):
    filename = get_path(args, 'checkpoint_{:04d}.pth.tar'.format(args.epochs-1))
    if not args.rerun_training and os.path.exists(filename): return filename

    train_enc_losses = []
    model.train()
    if args.train_predictor_individually:
        optimizer_encoder = optimizer[0]
        optimizer_pred = optimizer[1]
    else:
        optimizer_encoder = optimizer

    for epoch in range(args.start_epoch, args.epochs):
        if epoch >= args.early_stop:
            break
        if not args.fix_lr:
            adjust_learning_rate(optimizer_encoder, init_lr, epoch, args.epochs, warmup_epochs=args.warmup_epochs, warmup_init_lr=args.warmup_lr)
        # train encoder for one epoch, and train predictor for several epochs
        train_enc_loss = train_loop(train_loader_ssl, model, criterion, optimizer_encoder, epoch, args, device, logger, pred_only=False)
        train_enc_losses.append(train_enc_loss)
        if args.train_predictor_individually:
            for pred_epoch in range(args.pred_epochs):
                predictor_loss = train_loop(train_loader_ssl, model, criterion, optimizer_pred, pred_epoch, args, device, logger, pred_only=True)
        filename = get_path(args, 'checkpoint_{:04d}.pth.tar'.format(epoch))
        if (epoch + 1) % args.save_freq == 0 or epoch==args.epochs-1 or epoch==args.early_stop-1:
            checkpoint = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer_encoder.state_dict(),
                "loss": train_enc_loss
            }
            if args.train_predictor_individually:
                checkpoint['optimizer_pred'] = optimizer_pred.state_dict()
                checkpoint["predictor_loss"] = predictor_loss
            torch.save(checkpoint, filename)
            plt.plot(train_enc_losses, label = f"Training encoder loss")
            plt.legend()
            plt.savefig(get_path(args, 'training_encoder_loss.png'))
            plt.close()

        if epoch >= args.early_stop - 1:
            break
    return filename

def train_supervised(train_loader, model, criterion, optimizer, args, init_lr, device, logger, test_loader):
    filename = get_path(args, 'result.csv')
    if not args.rerun_training and os.path.exists(filename): return filename
    train_log_dict = {'epoch':[], 'train_acc':[], 'test_acc':[], 
                      'train loss':[], 'test loss':[]}
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args.epochs)
        train_loss, train_res = train_loop_supervised(train_loader, model, criterion, optimizer, epoch, args, device, logger)
        test_loss, test_res = validation_loop_supervised(test_loader, model, criterion, epoch, args, device, logger)
        filename = get_path(args, 'checkpoint_{:04d}.pth.tar'.format(epoch))
        if (epoch + 1) % 100 == 0 or epoch==args.epochs-1:
            checkpoint = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                "train loss": train_loss,
                "test loss": test_loss
            }
            checkpoint.update({f'train {k}': v for k, v in train_res.items()})
            checkpoint.update({f'test {k}': v for k, v in test_res.items()})
            torch.save(checkpoint, filename)
        for m in args.metrics:
            if f'train {m}' not in train_log_dict:
                train_log_dict[f'train {m}'] = []
                train_log_dict[f'test {m}'] = []
            train_log_dict[f'train {m}'].append(train_res[m])
            train_log_dict[f'test {m}'].append(test_res[m])
        train_log_dict['epoch'].append(epoch)
        train_log_dict['train loss'].append(train_loss)
        train_log_dict['test loss'].append(test_loss)
    train_log_df = pd.DataFrame.from_dict(data=train_log_dict).set_index('epoch')
    train_log_df.to_csv(get_path(args, 'trainin_log.csv'))
    for m in args.metrics:
        lines = train_log_df[[f'train {m}', f'test {m}']].plot.line()
        plt.savefig(get_path(args, f'training_log_{m}.png'))
        plt.close()
    lines = train_log_df[['train loss', 'test loss']].plot.line()
    plt.savefig(get_path(args, 'training_log_loss.png'))
    plt.close()

    res_dict = {"train loss": [train_loss], "test loss": [test_loss]}
    res_dict.update({f'train {k}': [v] for k, v in train_res.items()})
    res_dict.update({f'test {k}': [v] for k, v in test_res.items()})
    res_df = pd.DataFrame.from_dict(res_dict)
    res_df.to_csv(get_path(args, 'result.csv'))
    return test_res

if __name__ == '__main__':
    main()
