import snoop
import torch
import torchsnooper
import numpy as np
# torchsnooper.register_snoop()

# print(-float("Inf"))
d = torch.tensor([1, 2])
print(d.view((-1, 1)))


# from gomoku import GomokuEnv
# from time import sleep
# from datetime import datetime
# from ai import BOARD_SIZE
# player_color = 'black'

# # default 'beginner' level opponent policy 'random'
# env = GomokuEnv(player_color=player_color,
#                 opponent='beginner', board_size=BOARD_SIZE)
# env.reset()
# for i in range(3):
#     observation, reward, done, info = env.step(i)
#     env.render()
