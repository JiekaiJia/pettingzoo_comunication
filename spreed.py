import copy
import json

from pettingzoo.mpe import simple_spread_v2, simple_speaker_listener_v3
import torch
#
# from agent import CNetAgent
# from dotdic import DotDic
# from mpe_game import MpeGame
# from switch_cnet import SwitchCNet
# from arena import Arena
#
#
# # configure opts for Switch game with 3 DIAL agents
# def init_action_and_comm_bits(opt):
#     opt.comm_enabled = opt.game_comm_bits > 0 and opt.game_nagents > 1
#     if opt.model_comm_narrow is None:
#         opt.model_comm_narrow = opt.model_dial
#     if not opt.model_comm_narrow and opt.game_comm_bits > 0:
#         opt.game_comm_bits = 2 ** opt.game_comm_bits
#     if opt.comm_enabled:
#         opt.game_action_space_total = opt.game_action_space + opt.game_comm_bits
#     else:
#         opt.game_action_space_total = opt.game_action_space
#     return opt
#
#
# def init_opt(opt):
#     if not opt.model_rnn_layers:
#         opt.model_rnn_layers = 2
#     if opt.model_avg_q is None:
#         opt.model_avg_q = True
#     if opt.eps_decay is None:
#         opt.eps_decay = 1.0
#     opt = init_action_and_comm_bits(opt)
#     return opt
#
#
# def create_game(env, opt):
#     game_name = opt.game.lower()
#     if game_name == 'switch':
#         return MpeGame(env, opt)
#     else:
#         raise Exception('Unknown game: {}'.format(game_name))
#
#
# def create_agents(opt, game):
#     agents = [None]  # 1-index agents
#     cnet = create_cnet(opt)
#     cnet_target = copy.deepcopy(cnet)
#     for i in range(1, opt.game_nagents + 1):
#         agents.append(CNetAgent(opt, game=game, model=cnet, target=cnet_target, index=i))
#         if not opt.model_know_share:
#             cnet = create_cnet(opt)
#             cnet_target = copy.deepcopy(cnet)
#     return agents
#
#
# def create_cnet(opt):
#     game_name = opt.game.lower()
#     if game_name == 'switch':
#         return SwitchCNet(opt)
#     else:
#         raise Exception('Unknown game: {}'.format(game_name))
#
#
env = simple_spread_v2.env()
# opt = DotDic(json.loads(open('./switch_3_rial.json', 'r').read()))
# opt = init_opt(opt)
# game = create_game(env, opt)
# agents = create_agents(opt, game)
# arena = Arena(opt, game)

#comm = all_comm[:, torch.arange(all_comm.size(0)) != agent_idx]
# import supersuit
# a = torch.rand((2, 3, 4))
# print(a)
# print(a[(torch.arange(a.shape[0])) != 1])
# print(a.shape)
# m = supersuit.aec_wrappers.pad_observations(simple_speaker_listener_v3.env())

# env = simple_spread_v2.parallel_env()
env.reset()
# print(env.action_spaces)
# print(a[:, 2].zero_())
# print(a)

for agent in env.agent_iter():

    _, reward, done, _ = env.last()
    if not done:
        action = env.action_spaces[agent].sample()
    else:
        action = None

    print(f'{agent}: {reward}')

    env.step(action)
