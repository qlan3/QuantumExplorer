from agents.SAC import *
from components.qnn import *


class QuantumSAC(SAC):
  '''
  Implementation of Quantum SAC (Soft Actor-Critic)
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

  def createNN(self, input_type):
    # Set feature network
    feature_net = nn.Identity()
    # Set actor network
    assert self.action_type == 'CONTINUOUS', f'{self.agent_name} only supports continous action spaces.'
    # Use quantum actor network
    actor_net = QuantumSquashedGaussianActor(state_size=self.state_size, action_size=self.action_size, qnn_layers=self.cfg['qnn_layers'], qnn_type=self.cfg['qnn_type'], action_lim=self.action_lim, QPU_device=self.cfg['QPU_device'], torch_device=self.cfg['device'], rsample=True)
    # Set critic network
    critic_net = MLPDoubleQCritic(layer_dims=[self.state_size+self.action_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set the model
    NN = ActorDoubleQCriticNet(feature_net, actor_net, critic_net)
    return NN