# -*- coding: utf-8 -*
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
  def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
    super(Attention, self).__init__(**kwargs)
    
    self.supports_masking = True
    self.bias = bias
    self.feature_dim = feature_dim
    self.step_dim = step_dim
    self.features_dim = 0
    
    weight = torch.zeros(feature_dim, 1)
    nn.init.xavier_uniform_(weight)
    self.weight = nn.Parameter(weight)
    
    if bias:
        self.b = nn.Parameter(torch.zeros(1))
        
  def forward(self, x, mask=None):

    feature_dim = self.feature_dim
    step_dim = self.step_dim

    eij = torch.mm(
        x.contiguous().view(-1, feature_dim), 
        self.weight
    ).view(-1, step_dim)
    
    if self.bias:
        eij = eij + self.b
        
    eij = torch.tanh(eij)
    a = torch.exp(eij)
    
    if mask is not None:
        a = a * mask

    a = a / torch.sum(a, 1, keepdim=True) + 1e-10

    weighted_input = x * torch.unsqueeze(a, -1)
    return torch.sum(weighted_input, 1)

class SpatialDropout(nn.Module):

  def __init__(self,p):
    super(SpatialDropout, self).__init__()
    self.dropout = nn.Dropout2d(p)

  def forward(self, x):

      x = x.permute(0, 2, 1)   # convert to [batch, feature, timestep]
      x = self.dropout(x)
      x = x.permute(0, 2, 1)   # back to [batch, timestep, feature]
      return x

class NeuralNet(nn.Module):

  def __init__(self, embedding_matrix, num_unit, num_layers=1):
    super(NeuralNet, self).__init__()
    self.max_feature = embedding_matrix.shape[0]
    self.embedding_size = embedding_matrix.shape[1]
    self.embedding = nn.Embedding(self.max_feature, self.embedding_size)
    self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
    self.embedding.weight.requires_grad = False
    self.embedding_dropout = SpatialDropout(0.3)
    self.lstm1 = nn.LSTM(self.embedding_size, num_unit, num_layers=num_layers, bidirectional=True, batch_first=True)
    self.lstm2 = nn.LSTM(num_unit*2, int(num_unit/2), num_layers=num_layers, bidirectional=True, batch_first=True)
    self.attention = Attention(num_unit, MAX_LEN)
    self.linear1 = nn.Linear(num_unit*3, num_unit)
    self.linear2 = nn.Linear(num_unit*3, num_unit)
    self.linear_out = nn.Linear(num_unit, 1)
    self.linear_aux_out = nn.Linear(num_unit, 5)

  def forward(self, x):

    h_embedding = self.embedding(x)
    h_embedding = self.embedding_dropout(h_embedding)
    h_lstm1, _ = self.lstm1(h_embedding)
    h_lstm2, _ = self.lstm2(h_lstm1)

    # attention
    att = self.attention(h_lstm2)

    # global average pooling
    avg_pool = torch.mean(h_lstm2, 1)

    # global max pooling
    max_pool, _ = torch.max(h_lstm2, 1)

    # concatenation
    h = torch.cat((max_pool, avg_pool, att), 1)

    h_linear1 = F.relu(self.linear1(h))
    h_linear2 = F.relu(self.linear2(h))

    out1 = F.sigmoid(self.linear_out(h_linear1))
    out2 = F.sigmoid(self.linear_aux_out(h_linear2))

    return out1, out2


### test code
if __name__ == '__main__':
  MAX_LEN = 220
  embedding_matrix = np.random.random([1000,300])
  net = NeuralNet(embedding_matrix, 128)
  print(net)


