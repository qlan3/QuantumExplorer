import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import pennylane as qml


def layer_init(layer, init_type='default', nonlinearity='relu', w_scale=1.0):
  nonlinearity = nonlinearity.lower()
  # Initialize all weights and biases in layer and return it
  if init_type in ['uniform_', 'normal_']:
    getattr(nn.init, init_type)(layer.weight.data)
  elif init_type in ['xavier_uniform_', 'xavier_normal_', 'orthogonal_']:
    # Compute the recommended gain value for the given nonlinearity
    gain = nn.init.calculate_gain(nonlinearity)
    getattr(nn.init, init_type)(layer.weight.data, gain=gain)
  elif init_type in ['kaiming_uniform_', 'kaiming_normal_']:
    getattr(nn.init, init_type)(layer.weight.data, mode='fan_in', nonlinearity=nonlinearity)
  else: # init_type == 'default'
    return layer
  layer.weight.data.mul_(w_scale)
  nn.init.zeros_(layer.bias.data)
  return layer


def GetVQC(n_qubits, qnn_layers, qnn_type):
  if qnn_type == 'ReUploadingVQC':
    def ReUploadingVQC(inputs, entangling_weights, embedding_weights):
      '''
      A variational quantum circuit (VQC) with data re-uploading
      '''
      # Prepare all zero state
      all_zero_state = torch.zeros(n_qubits)
      qml.BasisStatePreparation(all_zero_state, wires=range(n_qubits))
      for i in range(qnn_layers):
        # Variational layer
        qml.StronglyEntanglingLayers(entangling_weights[i], wires=range(n_qubits))
        # Encoding layer
        features = inputs * embedding_weights[i]
        qml.AngleEmbedding(features=features, wires=range(n_qubits))
      # Last varitional layer
      qml.StronglyEntanglingLayers(entangling_weights[-1], wires=range(n_qubits))
      return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    # Get weight shape
    entangling_weights_shape = (qnn_layers+1, ) + qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n_qubits)
    embedding_weights_shape = (qnn_layers, n_qubits)
    weight_shapes = {
      'entangling_weights': entangling_weights_shape,
      'embedding_weights': embedding_weights_shape
    }
    return ReUploadingVQC, weight_shapes
  elif qnn_type == 'NormalVQC':
    def NormalVQC(inputs, entangling_weights):
      '''
      A variational quantum circuit (VQC) (without data re-uploading)
      '''
      qml.AngleEmbedding(features=inputs, wires=range(n_qubits))
      qml.StronglyEntanglingLayers(entangling_weights, wires=range(n_qubits))
      return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    entangling_weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=qnn_layers, n_wires=n_qubits)
    weight_shapes = {'entangling_weights': entangling_weights_shape}
    return NormalVQC, weight_shapes


class HybridModel(torch.nn.Module):
  def __init__(self, input_dim, output_dim, qnn_layers, qnn_type, last_w_scale, QPU_device, torch_device):
    super().__init__()
    # Create a QNode
    n_qubits = input_dim # the number of qubits is the same as the input dimension
    if QPU_device == 'default.qubit.torch':
      dev = qml.device(QPU_device, wires=n_qubits, torch_device=torch_device)
    else:
      dev = qml.device(QPU_device, wires=n_qubits)
    VQC, weight_shapes = GetVQC(n_qubits, qnn_layers, qnn_type)
    qnode = qml.QNode(VQC, dev, interface='torch', diff_method='best')
    self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    # Create a output layer
    self.output = layer_init(nn.Linear(input_dim, output_dim), init_type='kaiming_uniform_', w_scale=last_w_scale)

  def forward(self, x):
    x = self.qlayer(x)
    x = self.output(x)
    return x


class QuantumSquashedGaussianActor(nn.Module):
  def __init__(self, state_size, action_size, qnn_layers, qnn_type, action_lim, last_w_scale=1e-3, QPU_device='default.qubit', torch_device='cpu', rsample=False):
    super().__init__()
    self.rsample = rsample
    self.actor_net = HybridModel(input_dim=state_size, output_dim=2*action_size, qnn_layers=qnn_layers, qnn_type=qnn_type, last_w_scale=last_w_scale, QPU_device=QPU_device, torch_device=torch_device)
    self.action_lim = action_lim

  def distribution(self, phi):
    action_mean, action_std = self.actor_net(phi).chunk(2, dim=-1)
    # Constrain action_std inside [1e-6, 10]
    action_std = torch.clamp(F.softplus(action_std), 1e-6, 10)
    return action_mean, action_std, Normal(action_mean, action_std)

  def log_pi_from_distribution(self, action_distribution, action):
    # NOTE: Check out the original SAC paper and https://github.com/openai/spinningup/issues/279 for details
    log_pi = action_distribution.log_prob(action).sum(axis=-1)
    log_pi -= (2*(math.log(2) - action - F.softplus(-2*action))).sum(axis=-1)
    # Constrain log_pi inside [-20, 20]
    log_pi = torch.clamp(log_pi, -20, 20)
    return log_pi

  def forward(self, phi, action=None, deterministic=False):
    # Compute action distribution and the log_pi of given actions
    action_mean, action_std, action_distribution = self.distribution(phi)
    if action is None:
      if deterministic:
        u = action_mean
      else:
        u = action_distribution.rsample() if self.rsample else action_distribution.sample()
      action = self.action_lim * torch.tanh(u)
    else:
      u = torch.clamp(action / self.action_lim, -0.999, 0.999)
      u = torch.atanh(u)
    # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
    log_pi = self.log_pi_from_distribution(action_distribution, u)
    return action, action_mean, action_std, log_pi