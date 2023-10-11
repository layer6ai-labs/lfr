import torch
import torch.nn as nn

def random_permanent_dropout(model):
    model.apply(randomly_set_to_zero) #M

def randomly_set_to_zero(m):
    with torch.no_grad():
        if type(m)==nn.Linear or type(m)==nn.Conv2d or type(m)==nn.Conv1d:
            n = m.weight.numel()
            drop_num = int(round(n*0.4))
            indices = torch.randperm(n)[:drop_num]
            m.weight = m.weight.contiguous()  
            m.weight.flatten()[indices] = 0

def initialize_conv_with_beta(model):
    model.apply(initialize_layer_with_beta) #M

def initialize_layer_with_beta(m):
    with torch.no_grad():
        if type(m)==nn.Conv2d:
            # sample weights intialization from beta distribution
            beta_dist = torch.distributions.beta.Beta(torch.tensor([0.5]), 
                        torch.tensor([0.5]))
            weight_size = m.weight.size()
            # scale to [-1,1]
            random_weights = 2*beta_dist.sample(weight_size)-1
            m.weight.data = random_weights.view(weight_size)

def regularize_model_parameters(model):
    model.apply(regularize_parameters) #M

def regularize_parameters(m):
    with torch.no_grad():
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            for param in m.parameters():
                param.data = param.data/torch.norm(param.data)


def remove_bias_terms(model):
    model.apply(zero_bias) #M

def zero_bias(m):
    with torch.no_grad():
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            m.bias.data.zero_()
