from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from linear_eval import report_auprc, report_auroc, report_kappa


def train_loop(train_loader, model, criterion, optimizer, epoch, args, device, logger, pred_only=False):
    if args.method == 'lfr':
        return train_loop_lfr(train_loader, model, criterion, optimizer, epoch, args, device, logger, pred_only)
    elif 'diet' in args.method:
        return train_loop_diet(train_loader, model, criterion, optimizer, epoch, args, device, logger)
    elif args.method == 'simsiam':
        return train_loop_simsiam(train_loader, model, criterion, optimizer, epoch, args, device, logger)
    elif args.method == 'stab':
        return train_loop_stab(train_loader, model, criterion, optimizer, epoch, args, device, logger)
    elif args.method == 'simclr':
        return train_loop_simclr(train_loader, model, criterion, optimizer, epoch, args, device, logger)
    elif args.method == 'autoencoder':
        return train_loop_ae(train_loader, model, criterion, optimizer, epoch, args, device, logger)
    else: 
        raise NotImplementedError


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def bt_loss_bs(p, z, lambd=0.01, normalize=False):
    #barlow twins loss but in batch dims
    c = torch.matmul(F.normalize(p), F.normalize(z).T)
    assert c.min()>-1 and c.max()<1
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    if normalize: loss = loss/p.shape[0]
    return loss


def train_loop_lfr(train_loader, model, criterion, optimizer, epoch, args, device, logger, pred_only):
    # switch to train mode
    model.train()
    model.target_encoders.eval()
    if pred_only: model.online_encoder.eval()
    train_loss = []
    for i, data in enumerate(train_loader):
        if args.train_with_steps and i >= 1: break
        if len(data) == 2: 
            x = data[0]
            x = x.to(device, non_blocking=True, dtype=torch.float)
        else: 
            x = data
            x = x.to(device)
        # compute output and loss
        predicted_reps, target_reps = model(x)
        loss = torch.tensor(0).to(device)
        for t in range(args.num_targets):
            p = predicted_reps[t]
            z = target_reps[t]
            if args.loss == 'cosine':
                loss = loss - criterion(p, z).mean()
            elif args.loss == 'barlow-batch':
                loss = loss + bt_loss_bs(p,z, lambd=args.lambd)
            else:
                raise NotImplementedError
        loss = loss/args.num_targets 
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().item())
    train_loss = np.mean(train_loss)
    if pred_only:
        logger.info('Training predictor | Epoch: {} | loss: {}' .format(epoch, train_loss))
    else:
        logger.info('Training | Epoch: {} | loss: {}' .format(epoch, train_loss))
    return train_loss


def train_loop_stab(train_loader, model, criterion, optimizer, epoch, args, device, logger):
    # reference: https://table-representation-learning.github.io/assets/papers/stab_self_supervised_learning_.pdf
    model.train()
    train_loss = []
    for i, (images, _) in enumerate(train_loader):
        if args.train_with_steps and i > 1: break
        x = images.to(device, non_blocking=True, dtype=torch.float)
        p1, p2, z1, z2 = model(x)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().item())
    train_loss = np.mean(train_loss)
    logger.info('Training | Epoch: {} | loss: {}' .format(epoch,  train_loss))
    return train_loss


def train_loop_simsiam(train_loader, model, criterion, optimizer, epoch, args, device, logger):
    # adapted from https://github.com/facebookresearch/simsiam
    # switch to train mode
    model.train()
    train_loss = []
    for i, (images, _) in enumerate(train_loader):
        if args.train_with_steps and i >= 1: break
        x1 = images[0].to(device, non_blocking=True, dtype=torch.float)
        x2 = images[1].to(device, non_blocking=True, dtype=torch.float)
        # compute output and loss
        p1, p2, z1, z2 = model(x1, x2)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().item())
    train_loss = np.mean(train_loss)
    logger.info('Training | Epoch: {} | loss: {}' .format(epoch,  train_loss))
    return train_loss


def train_loop_simclr(train_loader, model, criterion, optimizer, epoch, args, device, logger):
    # adapted from https://github.com/sthalles/SimCLR/blob/master/simclr.py
    # switch to train mode
    model.train()
    train_loss = []
    for i, (images, _) in enumerate(train_loader):
        if args.train_with_steps and i >= 1: break
        x1 = images[0].to(device, non_blocking=True, dtype=torch.float)
        x2 = images[1].to(device, non_blocking=True, dtype=torch.float)
        # compute output and loss
        z1, z2 = model(x1, x2)
        logits, labels = model.info_nce_loss(z1, z2, device, temperature=0.07)
        loss = criterion(logits, labels)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().item())
    train_loss = np.mean(train_loss)
    return train_loss


def train_loop_ae(train_loader, model, criterion, optimizer, epoch, args, device, logger):
    # switch to train mode
    model.train()
    train_loss = []
    for i, data in enumerate(train_loader):
        if args.train_with_steps and i >= 1: break
        if len(data) == 2: 
            x = data[0]
            x = x.to(device, non_blocking=True, dtype=torch.float)
        else: 
            x = data
            x = x.to(device)
        # compute output and loss
        x_recon = model(x)
        loss = criterion(x_recon, x)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().item())
    train_loss = np.mean(train_loss)
    logger.info('Training | Epoch: {} | loss: {}' .format(epoch, train_loss))
    return train_loss


def train_loop_diet(train_loader, model, criterion, optimizer, epoch, args, device, logger):
    # switch to train mode
    model.train()
    train_loss = []
    for i, (idx, data) in enumerate(train_loader):
        if args.train_with_steps and i >= 1: break
        data = data.to(device, non_blocking=True, dtype=torch.float)
        idx = idx.to(device, non_blocking=True, dtype=torch.int64)
        _, pred = model(data)
        loss = criterion(pred, idx)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().item())
    train_loss = np.mean(train_loss)
    logger.info('Training | Epoch: {} | loss: {}' .format(epoch,  train_loss))
    return train_loss


def train_loop_supervised(train_loader, model, criterion, optimizer, epoch, args, device, logger):
    # switch to train mode
    model.train()
    train_loss = []
    train_total = 0
    train_correct = 0
    labels = []
    predictions = []
    test_metrics = {m: [] for m in args.metrics}
    res = {}
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, non_blocking=True, dtype=torch.float), targets.to(device, dtype=torch.int64)
        # compute output and loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().item())

        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()
        labels += targets.detach().tolist()
        predictions += F.softmax(outputs, dim=-1).detach().tolist()
    train_loss = np.mean(train_loss)
    train_acc = train_correct/train_total
    res['acc'] = train_acc
    labels, predictions = np.array(labels), np.array(predictions)
    if 'auprc' in test_metrics:
        auprc = report_auprc(labels, predictions[:, 1])
        res['auprc'] = auprc
    if 'auroc' in test_metrics:
        auroc = report_auroc(labels, predictions[:, 1])
        res['auroc'] = auroc
    if 'kappa' in test_metrics:
        kappa = report_kappa(labels, np.argmax(predictions, axis=1))
        res['kappa'] = kappa
    logger.info('Training | Epoch: {} | loss: {}' .format(epoch, train_loss) + ''.join([f' | {m}: {v}' for m, v in res.items()]))
    logger.info(f'confusion matrix: {metrics.confusion_matrix(labels, np.argmax(predictions, axis=1))}')
    return train_loss, res


def validation_loop_supervised(test_loader, model, criterion, epoch, args, device, logger):
    model.eval()
    test_loss = []
    test_correct = 0
    test_total = 0
    labels = []
    predictions = []
    test_metrics = {m: [] for m in args.metrics}
    res = {}

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device, non_blocking=True, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss.append(loss.detach().item())
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            labels += targets.detach().tolist()
            predictions += F.softmax(outputs, dim=-1).detach().tolist()
    
    test_loss = np.mean(test_loss)
    test_acc = test_correct/test_total
    res['acc'] = test_acc
    labels, predictions = np.array(labels), np.array(predictions)
    if 'auprc' in test_metrics:
        auprc = report_auprc(labels, predictions[:, 1])
        res['auprc'] = auprc
    if 'auroc' in test_metrics:
        auroc = report_auroc(labels, predictions[:, 1])
        res['auroc'] = auroc
    if 'kappa' in test_metrics:
        kappa = report_kappa(labels, np.argmax(predictions, axis=1))
        res['kappa'] = kappa
    logger.info('Testing | Epoch: {} | loss: {}' .format(epoch, test_loss) + ''.join([f' | {m}: {v}' for m, v in res.items()]))
    return test_loss, res
