from ssl_models.models.encoders import *
from ssl_models.models.decoders import *
from ssl_models.models.targets import *
from utils.model_utils import initialize_conv_with_beta, \
random_permanent_dropout, regularize_model_parameters, remove_bias_terms

def build_encoder(args, supervised=False):
    if args.dataset == 'uci-income':
        assert args.arch == 'mlp'
        return Simple4LayerMLP(105, dim=args.dim)
    elif args.dataset == 'theorem':
        assert args.arch == 'mlp'
        return Simple4LayerMLP(51, dim=args.dim)
    elif args.dataset == 'hepmass':
        assert args.arch == 'mlp'
        return Simple4LayerMLP(27, dim=args.dim)
    elif args.dataset == 'epilepsy':
        assert args.arch == 'cnn'
        return EpilepsyCnnEncoder(dim=args.dim)
    elif args.dataset == 'har':
        assert args.arch == 'cnn'
        return HARSCnnEncoder(dim=args.dim)
    elif args.dataset == 'kvasir':
        if args.arch == 'cnn':
            return KvasirCnnEncoder(dim=args.dim)
        elif 'resnet' in args.arch:
            return KvasirResnetEncoder(args.arch, dim=args.dim, args=args)
        else:
            raise ValueError(f"Encoder architecture of {args.arch} not found for dataset {args.dataset}")
    elif args.dataset == 'mimic3-los':
        assert args.arch == 'tcn'
        return MIMIC3TcnEncoder(42, 48, embedding_size=args.dim)
    else:
        raise ValueError(f"Encoder architecture not found for dataset {args.dataset}")


def build_target(args, num_of_layers, train_data=None, device=None):
    if args.dataset == 'uci-income':
        assert num_of_layers == 2 
        model = Simple2LayerMLP(input_dim=105, dim=args.dim)
    elif args.dataset == 'theorem':
        assert num_of_layers == 2 
        model = Simple2LayerMLP(input_dim=51, dim=args.dim)
    elif args.dataset == 'hepmass':
        assert args.arch == 'mlp'
        model = Simple2LayerMLP(27, dim=args.dim)
    elif args.dataset == 'epilepsy':
        model = EpilepsyCnnTarget(dim=args.dim)
    elif args.dataset == 'har':
        model = HARCnnTarget(dim=args.dim)
    elif args.dataset == 'kvasir':
        model = KvasirCnnTarget(dim=args.dim)
    elif args.dataset == 'mimic3-los':
        model = MIMIC3TcnTarget(42, 48, output_dim=args.dim)
    else:
        raise ValueError(f"Target architecture not found for dataset {args.dataset}")
    
    if args.random_dropout:
        random_permanent_dropout(model)
    if args.init_beta:
        initialize_conv_with_beta(model)
    if args.regularize_weight:
        regularize_model_parameters(model)
    if args.no_bias:
        remove_bias_terms(model)
    model = model.to(device)
    return model


def build_decoder(args):
    # for autoencoders
    if args.dataset == 'uci-income':
        decoder = Simple2LayerMLP(input_dim=args.dim, dim=105)
    elif args.dataset == 'theorem':
        assert args.arch == 'mlp'
        decoder = Simple2LayerMLP(input_dim=args.dim, dim=51)
    elif args.dataset == 'hepmass':
        assert args.arch == 'mlp'
        decoder = Simple2LayerMLP(input_dim=args.dim, dim=27)
    elif args.dataset == 'har':
        return HARCnnDecoder(dim=args.dim)
    elif args.dataset == 'epilepsy':
        assert args.arch == 'cnn'
        return EpilepsyCnnDecoder(dim=args.dim)
    elif args.dataset == 'kvasir':
        return KvasirCNNDecoder(dim=args.dim)
    elif args.dataset == 'mimic3-los':
        return MIMIC3TcnDecoder(input_dim=args.dim, seq_len=48)
    return decoder


def build_predictor(dim, pred_dim, args):
    # for both simsiam and lfr
    # a simple linear layer
    if args.pred_layers == 1:
        return nn.Linear(dim, dim)   
    # build a 2-layer predictor 
    elif args.pred_layers == 2:
        return nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                            nn.BatchNorm1d(pred_dim),
                            nn.ReLU(inplace=True), # hidden layer
                            nn.Linear(pred_dim, dim)) # output layer
    # build a 3-layer predictor
    elif args.pred_layers == 3: 
        return nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                            nn.BatchNorm1d(pred_dim),
                            nn.ReLU(inplace=True), # hidden layer
                            nn.Linear(pred_dim, pred_dim, bias=False),
                            nn.BatchNorm1d(pred_dim),
                            nn.ReLU(inplace=True), # hidden layer
                            nn.Linear(pred_dim, dim)) # output layer
    else:
        return ValueError(f"Predictors not implemented with {args.pred_layers} layers.")


def build_projector(dim, proj_dim, args):
    if args.dataset in ['theorem', 'uci-income', 'hepmass']:
        assert args.proj_layers == 2
        return nn.Sequential(nn.Linear(dim, proj_dim, bias=False),
                        nn.ReLU(inplace=True), # hidden layer
                        nn.Linear(proj_dim, dim)) # output layer
    # for simsiam
    # a simple linear layer
    if args.proj_layers == 1:
        return nn.Linear(dim, dim)

    # build a 2-layer predictor
    elif args.proj_layers == 2:
        
        return nn.Sequential(nn.Linear(dim, proj_dim, bias=False),
                            nn.BatchNorm1d(proj_dim),
                            nn.ReLU(inplace=True), # hidden layer
                            nn.Linear(proj_dim, dim)) # output layer

    # build a 3-layer predictor
    elif args.proj_layers == 3: 
        return nn.Sequential(nn.Linear(dim, proj_dim, bias=False),
                            nn.BatchNorm1d(proj_dim),
                            nn.ReLU(inplace=True), # hidden layer
                            nn.Linear(proj_dim, proj_dim, bias=False),
                            nn.BatchNorm1d(proj_dim),
                            nn.ReLU(inplace=True), # hidden layer
                            nn.Linear(proj_dim, dim)) # output layer
    else:
        return ValueError(f"Projectors not implemented with {args.proj_layers} layers.")
