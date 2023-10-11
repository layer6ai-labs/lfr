import logging
import math
import os
import sys
from ssl_models import *
from ssl_models.models.model_helper import build_encoder
import torch
import torch.nn as nn


def load_model(args, device, sample_data=None, num_data=None):
    model = None
    if  args.method == 'lfr':
        model = LFR(args.dim, args.pred_dim, device=device, num_targets=args.num_targets, args=args, sample_data=sample_data)
        model = model.to(device)
    elif args.method == 'simsiam':
        model = SimSiam(args.dim, args.pred_dim, args=args)
        model = model.to(device)
    elif args.method == 'stab':
        model = STab(args.dim, args.pred_dim, args=args)
        model = model.to(device)
    elif args.method == 'simclr':
        model = SimCLR(args.dim, args.pred_dim, args=args)
        model = model.to(device)
    elif args.method == 'autoencoder':
        model = AutoEncoder(device=device, args=args)
    elif 'supervised' in args.method: #supervised-aug
        model = supervised(args).to(device)
    elif 'diet' in args.method:
        model = Diet(args, num_data).to(device)
    else:
        raise NotImplementedError('Method {} has not been implemented'.format(args.method))
    return model


def get_path(args, file_name):
    path = os.path.join(args.ckpt_path, args.dataset, args.arch,
           args.method, 'dim_{}_pred_dim_{}'.format(args.dim, args.pred_dim), 
           'epochs_{}_optimizer_{}_lr_{}_bs_{}_momentum_{}_wd_{}'\
            .format(args.epochs, args.optimizer_type, args.lr, args.batch_size, args.momentum, args.weight_decay),
            'pred_lr_scale_{}'.format(args.pred_lr_scale))
    if args.method == 'stab':
        dir_name = f'drop_out_rate_{args.stab_drop_rate}'
        path = os.path.join(path, dir_name)
    if args.method == 'lfr':
        dir_name = 'num_targets_{}_loss_{}_temp_{}_pred_layers_{}'\
            .format(args.num_targets, args.loss, args.temp, args.pred_layers)
        dir_name = '{}_{}'.format(dir_name, args.target_layers[0])
        if args.train_predictor_individually: dir_name = '{}_train_predictor_individually_{}'.format(dir_name, args.pred_epochs)
        if args.fix_lr: dir_name = '{}_fix_lr'.format(dir_name)
        if args.random_dropout: dir_name = '{}_random_dropout'.format(dir_name)
        if args.init_beta: dir_name = '{}_init_beta'.format(dir_name)
        if args.regularize_weight: dir_name = '{}_regularize_weight'.format(dir_name)
        if args.target_sample_ratio > 1: dir_name = '{}_target_sample_ratio_{}'.format(dir_name, args.target_sample_ratio)
        path = os.path.join(path, dir_name)
    path = os.path.join(path, str(args.run_id))
    if not os.path.exists(path): os.makedirs(path)
    path = os.path.join(path, file_name)
    return path


def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)-5.5s] %(message)s',
                                      datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_handler = logging.FileHandler(
        filename, mode='a')  # default is 'a' to append
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stdout logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    logging.getLogger('matplotlib.font_manager').disabled = True
    return logger


def adjust_learning_rate(optimizer, init_lr, epoch, epochs, warmup_epochs=0, warmup_init_lr=0):
    if epoch < warmup_epochs:
        cur_lr = warmup_init_lr + (init_lr - warmup_init_lr) * epoch / warmup_epochs
    elif warmup_epochs <= epoch <= epochs:
        cur_lr = init_lr * (1 + math.cos((epoch - warmup_epochs) * math.pi /
                                         (epochs - warmup_epochs))) / 2
    else:
        raise ValueError('Step ({}) > total number of steps ({}).'.format(epoch, epochs ))
    
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            # do not change the learning rate
            param_group['lr'] = param_group['lr']
        else:
            param_group['lr'] = cur_lr


def get_optimizer(args, paramters, lr, weight_decay):
    if args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(paramters, lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    elif args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(paramters, lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(paramters, lr, momentum=args.momentum, weight_decay=weight_decay)
    return optimizer


def init_optimizer(model, args):
    # infer learning rate before changing batch size
    # init_lr = args.lr * args.batch_size / 256
    init_lr = args.lr
    optimizer = None
    if args.method == 'lfr':
        optimizer = init_optimizer_lfr(model, init_lr=init_lr, args=args)
    elif args.method == 'simsiam':
        optimizer = init_optimizer_simsiam(model, init_lr=init_lr, args=args)
    else:
        optimizer = get_optimizer(args, model.parameters(), init_lr, args.weight_decay)
    return optimizer, init_lr


def init_optimizer_lfr(model, init_lr, args):
    optim_params_encoder = [{'params': model.online_encoder.parameters(), 'fix_lr': False}]
    optim_params_pred = []
    for predictor in model.predictors:
        p = {'params': predictor.parameters()}
        if args.fix_pred_lr:
            p['fix_lr'] = True
            p['lr'] = init_lr*args.pred_lr_scale
        optim_params_pred.append(p)
    if args.train_predictor_individually:
        optimizer_encoder = get_optimizer(args, optim_params_encoder, init_lr, args.weight_decay)
        optimizer_pred = get_optimizer(args, optim_params_pred, init_lr, args.weight_decay)
        optimizer = (optimizer_encoder, optimizer_pred)
    else:
        optim_params_all = optim_params_encoder + optim_params_pred
        optimizer = get_optimizer(args, optim_params_all, init_lr, args.weight_decay)
    return optimizer


def init_optimizer_simsiam(model, init_lr, args):
    if args.fix_pred_lr:
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = get_optimizer(args, optim_params, init_lr, args.weight_decay)
    print(optimizer)
    return optimizer


def supervised(args):
    args.dim = args.num_of_classes
    model = build_encoder(args=args, supervised=True)
    return model


def get_criterion(args, device):
    if args.loss == 'cosine':
        criterion = nn.CosineSimilarity(dim=1).to(device)
    elif args.loss == 'mse':
        criterion = nn.MSELoss().to(device)
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.loss == 'ce-smooth':
        criterion = nn.CrossEntropyLoss(label_smoothing=0.8).to(device)
    else:
        # will use custom loss such as barlow-bs
        criterion = None
    return criterion


def load_ckpt(args, optimizer, model, logger):      
    # optionally resume from a checkpoint
    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.train_predictor_individually:
            optimizer[0].load_state_dict(checkpoint['optimizer'])
            optimizer[1].load_state_dict(checkpoint['optimizer_pred'])
            optimizer = (optimizer[0], optimizer[1])
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))
