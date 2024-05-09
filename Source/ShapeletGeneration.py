import torch
import torch.nn as nn
import numpy as np
from helper_datasets import convert_to_bags_2
from set_seed import set_seed

set_seed(88)
class ShapeletGeneration3LN(nn.Module):

  def __init__(self, n_prototypes
    , bag_size
    , n_classes
    ,stride_ratio
    ,features_to_use_str="min,max,mean,std"
    ):
    features_to_use=features_to_use_str.split(",")
    super(ShapeletGeneration3LN, self).__init__()
    # Currently, the prototypes are initialized with torch.randn
    # which generates a tensor with the given shape, filled with random numbers from a standard normal distribution.

    self.prototypes = (torch.randn((1, n_prototypes, bag_size))).requires_grad_()

    #self.prototypes = (torch.randn((1, n_prototypes, bag_size)) * 0.001).requires_grad_()
    self.n_p = n_prototypes
    self.bag_size = bag_size
    self.N = n_classes
    self.stride_ratio = stride_ratio
    self.features_to_use = features_to_use

    input_size_for_linear_layer = len(features_to_use)* n_prototypes
    hidden_layer_size = input_size_for_linear_layer * 2
    self.linear_layer1 = torch.nn.Linear(input_size_for_linear_layer,
                                        hidden_layer_size,
                                        bias=True)
    self.relu1 = nn.ReLU()
    self.linear_layer2 = torch.nn.Linear(hidden_layer_size,
                                        hidden_layer_size,
                                        bias=True)
    self.relu2 = nn.ReLU()
    """self.linear_layer3 = torch.nn.Linear(hidden_layer_size,
                                         hidden_layer_size,
                                         bias=True)
    self.relu3 = nn.ReLU()"""
    self.linear_layer_for_output = torch.nn.Linear(hidden_layer_size,
                                        n_classes,
                                        bias=True)

    


  def pairwise_distances(self, x, y):
    x_norm = (x.norm(dim=2)[:, :, None]).float()
    y_t = y.permute(0, 2, 1).contiguous()
    y_norm = (y.norm(dim=2)[:, None])
    y_t = torch.cat([y_t] * x.shape[0], dim=0)
    #print("x_norm.dtype",x_norm.dtype)
    #print("y_norm.dtype",y_norm.dtype)
    #print("x.dtype",x.dtype)
    #print("y_t.dtype",y_t.dtype)
    dist = x_norm + y_norm - 2.0 * torch.bmm(x.float(), y_t)
    return torch.clamp(dist, 0.0, np.inf)

  def layer_norm(self, feature):
    mean = feature.mean(keepdim=True, dim=-1)
    var = ((feature - mean)**2).mean(dim=-1, keepdim=True)
    std = (var + 1e-5).sqrt()
    y = (feature - mean) / std
    return y

  def get_output_from_prototypes(self, batch_inp):
    dist = self.pairwise_distances(batch_inp, self.prototypes)
    # I want to use torch standard function
    #dist = F.pairwise_distance(batch_inp, self.prototypes)
    l_features = []
    if "min" in self.features_to_use:
        min_dist = self.layer_norm(dist.min(dim=1)[0])
        l_features.append(min_dist)
    if "max" in self.features_to_use:
        max_dist = self.layer_norm(dist.max(dim=1)[0])
        l_features.append(max_dist)

    if "mean" in self.features_to_use:
        mean_dist = self.layer_norm(dist.mean(dim=1))
        l_features.append(mean_dist)
    
    if "std" in self.features_to_use:
        std_dist = self.layer_norm(dist.std(dim=1))
        l_features.append(std_dist)
    
    all_features = torch.cat(l_features, dim=1)
    return all_features
  

  def forward(self, x):
    x_bags = convert_to_bags_2(x, self.bag_size,self.stride_ratio)
    #print(x_bags.dtype)

    all_features = self.get_output_from_prototypes(x_bags)
    out = self.linear_layer1(all_features)
    out = self.relu1(out)
    out = self.linear_layer2(out)
    out = self.relu2(out)
    out = self.linear_layer_for_output(out)
    return out

