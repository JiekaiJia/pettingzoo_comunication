""""""

from ray.rllib.agents.qmix.model import _get_size
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from torch import nn
import torch.nn.functional as F


class RialDial(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.obs_size = _get_size(obs_space)
        self.self.model_config = self.model_config
        self.rnn_hidden_dim = self.model_config.lstm_cell_size

        self.comm_size = self.model_config.game_comm_bits`
        self.init_param_range = (-0.08, 0.08)

        # Set up inputs
        self.agent_lookup = nn.Embedding(self.model_config.game_nagents, self.rnn_hidden_dim)
        # self.state_lookup = nn.Linear(18, self.rnn_hidden_dim)

        # Action aware
        self.prev_message_lookup = None
        if self.model_config.model_action_aware:
            if self.model_config.model_dial:
                self.prev_action_lookup = nn.Embedding(self.model_config.game_action_space_total, self.rnn_hidden_dim)
            else:
                self.prev_action_lookup = nn.Embedding(self.model_config.game_action_space + 1, self.rnn_hidden_dim)
                self.prev_message_lookup = nn.Embedding(self.model_config.game_comm_bits + 1, self.rnn_hidden_dim)

        # Communication enabled
        if self.model_config.comm_enabled:
            self.messages_mlp = nn.Sequential()
            if self.model_config.model_bn:
                self.messages_mlp.add_module('batchnorm1', nn.BatchNorm1d(self.comm_size))
            self.messages_mlp.add_module('linear1', nn.Linear(self.comm_size, self.rnn_hidden_dim))
            if self.model_config.model_comm_narrow:
                self.messages_mlp.add_module('relu1', nn.ReLU(inplace=True))

        # Set up RNN
        dropout_rate = self.model_config.model_rnn_dropout_rate or 0
        self.rnn = nn.GRU(input_size=self.rnn_hidden_dim, hidden_size=self.rnn_hidden_dim,
                          num_layers=self.model_config.model_rnn_layers, dropout=dropout_rate, batch_first=True)

        # Set up outputs
        self.outputs = nn.Sequential()
        if dropout_rate > 0:
            self.outputs.add_module('dropout1', nn.Dropout(dropout_rate))
        self.outputs.add_module('linear1', nn.Linear(self.rnn_hidden_dim, self.rnn_hidden_dim))
        if self.model_config.model_bn:
            self.outputs.add_module('batchnorm1', nn.BatchNorm1d(self.rnn_hidden_dim))
        self.outputs.add_module('relu1', nn.ReLU(inplace=True))
        self.outputs.add_module('linear2', nn.Linear(self.rnn_hidden_dim, self.model_config.game_action_space_total))
        
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [self.fc1.weight.new(1, self.rnn_hidden_dim).zero_().squeeze(0)]
        return h

    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def forward_rnn(self, input_dict, state, seq_lens):
        """
        Args:
            input_dict(dict): dictionary of input tensors, including “obs”, “obs_flat”, “prev_action”, “prev_reward”, 
            “is_training”, “eps_id”, “agent_id”, “infos”, and “t”.
            state: list of state tensors with sizes matching those returned by get_initial_state + the batch dimension.
            seq_lens: 1d tensor holding input sequence lengths
        """
        s_t = Variable(s_t)
        hidden = Variable(hidden)
        prev_message = None
        if self.model_config.model_dial:
            if self.model_config.model_action_aware:
                prev_action = Variable(prev_action)
        else:
            if self.model_config.model_action_aware:
                prev_action, prev_message = input_dict['prev_action']
                prev_action = Variable(prev_action)
                prev_message = Variable(prev_message)
            messages = Variable(messages)
        agent_index = Variable(agent_index)

        z_a, z_o, z_u, z_m = [0] * 4
        z_a = self.agent_lookup(agent_index)
        z_o = s_t

        if self.model_config.model_action_aware:
            z_u = self.prev_action_lookup(prev_action)
            if prev_message is not None:
                z_u += self.prev_message_lookup(prev_message)
        z_m = self.messages_mlp(messages.view(-1, self.comm_size))

        z = z_a + z_o + z_u + z_m
        z = z.unsqueeze(1)

        rnn_out, h_out = self.rnn(z, hidden)
        outputs = self.outputs(rnn_out[:, -1, :].squeeze())
        return q, [h]




