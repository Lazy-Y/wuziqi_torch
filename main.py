from gomoku import GomokuEnv
from time import sleep
from datetime import datetime
from ai import BOARD_SIZE, WuziGo, coord_to_action
import numpy as np
player_color = 'black'

# default 'beginner' level opponent policy 'random'
debug = False
train = True
env = GomokuEnv(player_color=player_color,
                opponent='beginner', board_size=BOARD_SIZE)
ai = WuziGo()
ai.load()
for epoch in range(10):
    total = 100
    win, draw, lose = 0, 0, 0
    for step in range(total):
        env.reset()
        observation = env.state.board.board_state
        observation = np.where(observation == 2, -1, observation)
        while True:
            action = ai.play(observation)
            observation, reward, done, info = env.step(action)
            observation = np.where(observation == 2, -1, observation)
            coord = env.state.board.last_coord
            if reward == 1 or done:
                break
            action = coord_to_action(*coord)
            ai.observe(action, observation)
            if done:
                break
        if debug:
            env.render()
        if train:
            ai.reward(reward)
        if reward == 1:
            win += 1
        elif reward == -1:
            lose += 1
        else:
            draw += 1
        if debug:
            print('win:', win, 'drar:', draw, 'lose:', lose)
    print('epoch:', epoch, 'win:', win, 'drar:', draw, 'lose:', lose)
    ai.save(epoch)
