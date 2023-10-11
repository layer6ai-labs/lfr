import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import torch
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from ssl_models.lfr import build_encoder
from matplotlib import pyplot as plt
from utils.utils import adjust_learning_rate, get_optimizer


def knn_eval(z_train, z_test, y_train, y_test, logger):
    if logger: logger.info("=> Running knn evaluation")
    knn = KNeighborsClassifier(n_neighbors=200, metric='cosine')
    knn.fit(z_train, y_train)
    accuracy = knn.score(z_test, y_test)
    if logger: logger.info("Finished knn evaluation, acc: {:.4f}".format(accuracy))
    return accuracy


def logistic_regression_eval(z_train, y_train, z_test, y_test, logger=None, save_res=None, save_name=None):
    """Evaluates representations using Logistic Regression model.
    """
    train_metrics = {'acc': []}
    test_metrics = {'acc': []}
    if logger: logger.info('Running Logistic regression')
    clf = LogisticRegression(max_iter=100000, solver='lbfgs')
    clf.fit(z_train, y_train)
    train_acc = clf.score(z_train, y_train)
    test_acc = clf.score(z_test, y_test)
    if logger:
        logger.info(f'Finished Logistic regression, train acc {train_acc:.4f}, test acc {test_acc :.4f}')
    if save_res is not None:
        if not os.path.exists(save_res): os.mkdir(save_res)   
        res_df = pd.DataFrame.from_dict({'train_acc':[train_acc], "test_acc":[test_acc]})
        res_df.to_csv(f'{save_res}/{save_name}.csv')
    train_metrics['acc'] = train_acc
    test_metrics['acc'] = test_acc
    return train_metrics, test_metrics


def create_and_load_encoder(args, pretrained):
    print("=> creating model '{}'".format(args.arch))
    encoder = build_encoder(args=args)
    load_pretrained_encoder(encoder, pretrained, args.method)
    return encoder


def get_linear_eval_model(args):
    if args.dataset not in ['mimic3-los']:
        model = nn.Linear(args.dim, args.num_of_classes)
    else:
        model = nn.Sequential(nn.Linear(args.dim, args.dim), 
                            nn.LayerNorm(args.dim), 
                            nn.ReLU(), nn.Dropout(0), 
                            nn.Linear(args.dim, args.num_of_classes))
    return model


def train_eval(args, model, train_loader=None, val_loader=None, 
                 logger=None, save_res=None, save_name=None, device=None, finetune=False):
    if finetune:
        eval_epochs = args.finetune_epochs
        run_name = 'finetune'
        init_lr = args.finetune_lr
    else:
        eval_epochs = args.eval_epochs
        run_name = 'linear'
        init_lr = args.eval_lr
    if save_res is not None:
        if not os.path.exists(save_res): os.makedirs(save_res)
        filename = os.path.join(save_res, f'{run_name}_eval_{eval_epochs}.pth.tar')

    model = model.to(device)
    model.eval()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # optimize only the linear classifier
    optimizer = get_optimizer(args, model.parameters(), init_lr, weight_decay=args.eval_wd)

    cudnn.benchmark = True

    METRICS = args.metrics
    train_metrics = {m: [] for m in METRICS}
    test_metrics = {m: [] for m in METRICS}
    best_metrics = {m: 0 for m in METRICS}

    for epoch in range(eval_epochs):
        if not args.fix_eval_lr:
            adjust_learning_rate(optimizer, init_lr, epoch, eval_epochs)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, device, logger=logger, finetune=finetune)

        # evaluate on validation set
        test_res = validate(val_loader, model, criterion, device, METRICS, logger=logger)
        train_res = validate(train_loader, model, criterion, device, METRICS, logger=logger)
        for m, v in train_res.items():
            train_metrics[m].append(v)

        for m, v in test_res.items():
            test_metrics[m].append(v)
            # remember best acc
            best_metrics[m] = max(best_metrics[m], v)

    if save_res is not None:
        res_dict = {}
        for m in METRICS:
            res_dict[f'train_{m}'] = [train_res[m]]
            res_dict[f"test_{m}"] = [best_metrics[m]]
            res_dict[f"final_test_{m}"] = [test_res[m]]
        res_df = pd.DataFrame.from_dict(res_dict)
        res_df.to_csv(f'{save_res}/{save_name}.csv')
        ckpt = {'epoch': eval_epochs,
                'state_dict': model.state_dict()}
        for m in METRICS:
            ckpt[f'train_{m}'] = [train_res[m]]
            ckpt[f"test_{m}"] = [test_res[m]]
        torch.save(ckpt, filename)

        for m in METRICS:
            filename = os.path.join(save_res, f'{run_name}_eval_{m}_{save_name}.png'.format(eval_epochs))
            plt.plot(train_metrics[m], label = f"Train {m}")
            plt.plot(test_metrics[m], label = f"Test {m}")
            plt.legend()
            plt.savefig(filename)
            plt.close()
    if logger is not None: 
        logger.info(f"Finished {run_name} evaluation,")
        for m in METRICS:
            logger.info(f"train {m}  {train_res[m]:4f}, final test {m}: {test_res[m]:.4f}, best {m}: {best_metrics[m]:.4f}")
    return train_res, test_res


def report_auprc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    auprc = metrics.auc(recall, precision)
    return auprc


def report_auroc(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    auroc = metrics.auc(fpr, tpr)
    return auroc


def report_kappa(y, pred):
    kappa = metrics.cohen_kappa_score(y, pred, weights='linear')
    return kappa


def train(train_loader, model, criterion, optimizer, epoch, args, device, logger=None, finetune=False):
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if finetune:
        model.train()
    else:
        model.eval()
    losses = []
    total = 0
    pred_correct = 0
    for i, (x, label) in enumerate(train_loader):
        x = x.to(device, dtype=torch.float)
        label = label.type(torch.LongTensor).to(device)

        # compute output
        output = model(x)
        loss = criterion(output, label)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = output.max(1)
        total += label.size(0)
        pred_correct += predicted.eq(label).sum().item()
        losses.append(loss.detach().item())
        
    average_loss = np.mean(losses)
    acc = pred_correct/total
    if logger is not None and epoch % args.print_freq == 0:
        logger.info(f'Train | Epoch: {epoch}| loss: {average_loss:.4f} | Acc@1: {acc:6.3f} ')
    return average_loss, acc


def validate(val_loader, model, criterion, device, metrics_required=['acc'], logger=None):
    losses = []
    total = 0
    pred_correct = 0
    labels = []
    predictions = []
    res = {}

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (x, target) in enumerate(val_loader):
            x = x.to(device, dtype=torch.float)
            target = target.type(torch.LongTensor).to(device)

            # compute output
            output = model(x)
            loss = criterion(output, target)
            # measure accuracy and record loss
            _, predicted = output.max(1)
            total += target.size(0)
            pred_correct += predicted.eq(target).sum().item()
            losses.append(loss.detach().item())
            labels += target.detach().tolist()
            predictions += nn.functional.softmax(output, dim=-1).detach().tolist()
            
    average_loss = np.mean(losses)
    acc = pred_correct/total
    res['acc'] = acc
    labels, predictions = np.array(labels), np.array(predictions)
    if 'auprc' in metrics_required:
        auprc = report_auprc(labels, predictions[:, 1])
        res['auprc'] = auprc
    if 'auroc' in metrics_required:
        auroc = report_auroc(labels, predictions[:, 1])
        res['auroc'] = auroc
    if 'kappa' in metrics_required:
        kappa = report_kappa(labels, np.argmax(predictions, axis=1))
        res['kappa'] = kappa
    
    return res


def load_pretrained_encoder(model, pretrained, method):
    prefix = 'online_encoder'
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")
        # rename LFR pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only online encoder up to before the embedding layer
            if k.startswith(prefix) and not k.startswith('encoder.fc'):
                # remove prefix
                state_dict[k[len("{}.".format(prefix)):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg.missing_keys)
        assert len(msg.missing_keys) == 0
        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))


def get_embeddings(model, loader):
    labels = []
    data_embeddings = []
    for x, y in loader:
        x = x.to(torch.device('cuda'), dtype=torch.float)
        embeddings = model(x)
        data_embeddings.extend(embeddings.detach().cpu())
        labels.extend(y.detach().cpu().tolist())
    data_embeddings = np.array(torch.stack(data_embeddings))
    labels = np.array(labels)
    return data_embeddings, labels


def get_representations(args, device, pretrained, train_loader, val_loader, save_rep=None):
    model = create_and_load_encoder(args, pretrained)
    return get_representations_from_model(model, train_loader, val_loader, device, save_rep=save_rep)


def get_representations_from_model(model, train_loader, val_loader, device, save_rep=None):
    model.eval()
    model.to(device)
    with torch.no_grad():
        train_embeddings, train_labels = get_embeddings(model, train_loader)
        test_embeddings, test_labels = get_embeddings(model, val_loader)
    if save_rep is not None:
        if not os.path.exists(save_rep): os.mkdir(save_rep)
        np.save(f'{save_rep}/X_train', train_embeddings)
        np.save(f'{save_rep}/X_test', test_embeddings)
        np.save(f'{save_rep}/y_train', train_labels)
        np.save(f'{save_rep}/y_test', test_labels)
    return train_embeddings, test_embeddings, train_labels, test_labels


def create_reps_loader(args, train_reps, val_reps, train_labels, val_labels):
    train_data = create_data(train_reps, train_labels)
    val_data = create_data(val_reps, val_labels)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.eval_bs, shuffle=True,
                num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(val_data, batch_size=args.eval_bs,
                shuffle=False, num_workers=args.workers, pin_memory=True) 
    return train_loader, test_loader


def create_data(x, y):
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i]])
    return data


def eval(args, device, train_loader, test_loader, save_rep=None, save_res=None, save_name=None, logger=None, ckpt_path=None, model=None):
    if model is not None:
        train_reps, test_reps, y_train, y_test = get_representations_from_model(model, train_loader, test_loader, device, save_rep=save_rep)
    else:
        train_reps, test_reps, y_train, y_test = get_representations(args, device, ckpt_path, train_loader, test_loader, save_rep=save_rep)
    
    if logger is not None:
        logger.info("=> Running evaluation")
    if args.dataset in ['har', 'epilepsy', 'kvasir', 'mimic3-los']: 
        train_reps_loader, val_reps_loader = create_reps_loader(args, train_reps, test_reps, y_train, y_test)
        train_res, test_res = train_eval(args, get_linear_eval_model(args), train_reps_loader, val_reps_loader, logger=logger, \
                                         save_res=save_res, save_name=save_name, device=device)
    else:
        train_res, test_res = logistic_regression_eval(train_reps, y_train, test_reps, y_test, logger=logger, save_res=save_res, save_name=save_name)
    knn_acc = 0
    if args.dataset not in ['mimic3-los', 'hepmass']:
        knn_acc=knn_eval(train_reps, test_reps, y_train, y_test, logger)

    return train_res, test_res, knn_acc


def eval_ft(args, device, train_loader, test_loader, ckpt_path, save_res=None, save_name=None, logger=None):
    logger.info("=> Running finetuning evaluation")
    ssl_model = create_and_load_encoder(args, ckpt_path)
    ft_model = nn.Sequential(ssl_model, get_linear_eval_model(args))

    train_res, test_res = train_eval(args, ft_model, train_loader, test_loader, logger=logger, \
                                     save_res=save_res, save_name=save_name, device=device, finetune=True)
    return train_res, test_res