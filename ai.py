import numpy as np
import snoop
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsnooper
from torchsummary import summary

torchsnooper.register_snoop()
snoop.install(enabled=False)

BOARD_SIZE = 11
ACTIONS = list(range(BOARD_SIZE ** 2))

board_ones = torch.ones((BOARD_SIZE, BOARD_SIZE))
board_zeros = torch.zeros((BOARD_SIZE, BOARD_SIZE))
board_infs = torch.tensor([[float('Inf')] * BOARD_SIZE] * BOARD_SIZE)


def action_to_coord(a):
    return (a // BOARD_SIZE, a % BOARD_SIZE)


def coord_to_action(i, j):
    return i * BOARD_SIZE + j  # action index


def coord_rot90(i, j):
    return BOARD_SIZE - 1 - j, i


def coord_flip_row(i, j):
    return BOARD_SIZE - 1 - i, j


def coord_flip_col(i, j):
    return i, BOARD_SIZE - 1 - j


def coord_flip_both(i, j):
    return BOARD_SIZE - 1 - i, BOARD_SIZE - 1 - j


def expand_board(action, single_board_np):
    """
    Args:
        action (int): single action integer
        single_board_np (ndarray): shape (BS, BS)
    Returns:
        action (tensor): shape (16)
        board_tensor (tensor): shape (16, BS, BS)
    """
    single_board_tensor = torch.tensor(single_board_np)
    board_tensor_array = [
        single_board_tensor,
        torch.flip(single_board_tensor, [0]),
        torch.flip(single_board_tensor, [1]),
        torch.flip(single_board_tensor, [0, 1]),
    ]
    board_tensor_array = [[board_tensor,
                           torch.rot90(board_tensor, 1),
                           torch.rot90(board_tensor, 2),
                           torch.rot90(board_tensor, 3)]
                          for board_tensor in board_tensor_array]
    board_tensor_array = [i for j in board_tensor_array for i in j]
    action_coord = action_to_coord(action)
    action_coord_array = [action_coord,
                          coord_flip_row(*action_coord),
                          coord_flip_col(*action_coord),
                          coord_flip_both(*action_coord)]
    final_action_coord_array = []
    for action_coord in action_coord_array:
        final_action_coord_array.append(action_coord)
        action_coord = coord_rot90(*action_coord)
        final_action_coord_array.append(action_coord)
        action_coord = coord_rot90(*action_coord)
        final_action_coord_array.append(action_coord)
        action_coord = coord_rot90(*action_coord)
        final_action_coord_array.append(action_coord)
    action_array = [coord_to_action(*coord)
                    for coord in final_action_coord_array]
    return torch.tensor(action_array), torch.stack(board_tensor_array)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1_num_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2,
                      out_channels=self.conv1_num_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.LeakyReLU(),
        )
        self.conv2_num_channels = 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv1_num_channels,
                      out_channels=self.conv2_num_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.LeakyReLU(),
        )
        self.out = nn.Softmax(1)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())
        self.reset()

    def reset(self):
        pass

    @snoop()
    def forward(self, board):
        """
        Args:
            board ([ndarray / tensor shape (-1, BS, BS)]) each element can be -1, 0, 1
        Returns:
            tensor shape (-1, BS * BS)
        """
        if torch.is_tensor(board):
            board_tensor = board
        else:
            board_tensor = torch.tensor(board)
        self_board_tensor = torch.where(
            board_tensor == 1, board_ones, board_zeros)
        oppo_board_tensor = torch.where(
            board_tensor == -1, board_ones, board_zeros)
        x = torch.stack(
            [self_board_tensor, oppo_board_tensor], axis=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sum(x, dim=1)
        x = torch.where(board_tensor == 0, x, -board_infs)
        x = torch.reshape(x, (-1, BOARD_SIZE ** 2))
        x = self.out(x)
        return x

    @snoop
    def train(self, action, single_board_np):
        """
        Args:
            action (int): [0, BS * BS)
            single_board_np ([ndarray shape (BS, BS)])
        """
        action_tensor, board_tensor = expand_board(action, single_board_np)
        output_tensor = self(board_tensor)
        loss = self.criterion(output_tensor, action_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class WuziGo:

    def __init__(self):
        self.policy_network = PolicyNetwork()
        self.reset()
        self.PATH = 'weights/w4.pt'

    def save(self, epoch):
        torch.save(self.policy_network.state_dict(), f'weights/w_5{epoch}.pt')

    def load(self):
        try:
            self.policy_network.load_state_dict(torch.load(self.PATH))
        except FileNotFoundError:
            print('unable to load model')

    def reset(self):
        self.policy_network.reset()
        self.observations = []  # stores pairs (action, single_board_np)

    def play(self, single_board_np):
        if len(self.observations) % 2:
            single_board_np = -single_board_np
        board = np.reshape(single_board_np, (1, BOARD_SIZE, BOARD_SIZE))
        probs = self.policy_network(board)
        action = np.random.choice(ACTIONS, p=probs.detach().numpy()[0])
        self.observations.append((action, single_board_np))
        return action

    def observe(self, action, single_board_np):
        if len(self.observations) % 2:
            single_board_np = -single_board_np
        self.observations.append((action, single_board_np))

    def reward(self, reward):
        """
        Args:
            reward (int): -1, 0, 1
        """
        if reward == 0:
            return
        for action, single_board_np in self.observations[::-2]:
            self.policy_network.train(action, single_board_np)
        self.reset()


def test_input():
    board_np = (np.arange(BOARD_SIZE ** 2) % 3) - 1
    board_np = np.reshape(board_np, (1, BOARD_SIZE, BOARD_SIZE))
    print(board_np)
    ai = PolicyNetwork()
    out = ai(board_np)
    single_board_np = np.squeeze(board_np)
    ai.train(0, single_board_np)
    ai.train(0, single_board_np)
    ai.train(0, single_board_np)
    ai.train(0, single_board_np)
    # rot_board_np = np.rot90(board_np[0])
    # print(rot_board_np)
    # rot90_coord = coord_rot90(0, 0)
    # rot_board_np = np.reshape(rot_board_np, (BOARD_SIZE, BOARD_SIZE))
    # ai.train(coord_to_action(*rot90_coord), rot_board_np)


def test_util():
    board_np = np.arange(BOARD_SIZE ** 2)
    board_np = np.reshape(board_np, (BOARD_SIZE, BOARD_SIZE))
    action_tensor, board_tensor = expand_board(0, board_np)
    print(action_tensor.shape)
    print(board_tensor.shape)


def test_train():
    board_np = np.zeros((BOARD_SIZE, BOARD_SIZE))
    board_np[2, 2] = -1
    print(board_np)
    action = coord_to_action(1, 1)
    ai = PolicyNetwork()
    prev = ai(np.expand_dims(board_np, axis=0))
    prev = torch.reshape(prev, (1, BOARD_SIZE, BOARD_SIZE))
    for _ in range(100):
        ai.train(action, board_np)
    curr = ai(np.expand_dims(board_np, axis=0))
    curr = torch.reshape(curr, (1, BOARD_SIZE, BOARD_SIZE))
    print(prev)
    print(curr)
    print(curr - prev)
    board_np[2, 2] = 0
    board_np[1, 1] = -1
    new = ai(np.expand_dims(board_np, axis=0))
    print(new.detach().numpy())


def test_net():
    net = PolicyNetwork()
    summary(net, (BOARD_SIZE, BOARD_SIZE))


def test_ai():
    ai = WuziGo()
    board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for i in range(2):
        g = input("Go: ")
        coord = [int(n) for n in g.split(',')]
        action = coord_to_action(*coord)
        ai.observe(action, board)
        board = torch.tensor(board)
        board[coord[0], coord[1]] = 1
        print(board)
        if i >= 1:
            break
        action = ai.play(board)
        board = torch.tensor(board)
        print('action', action)
        coord = action_to_coord(action)
        print('coord', coord)
        board[coord[0], coord[1]] = -1
        print(board)

    # coords = [(2, 2), (1, 1), (1, 2), (0, 2), (2, 0), (2, 1)]
    # for i, coord in enumerate(coords):
    #     action = coord_to_action(*coord)
    #     ai.observe(action, board)
    #     board = np.array(board)
    #     board[coord] = (-1) ** i
    # for act, board in ai.observations:
    #     print(action_to_coord(act))
    #     print(board, "\n")
    print(ai.observations)
    ai.reward(1)


if __name__ == "__main__":
    # test_input()
    # test_train()
    # test_util()
    # test_net()
    test_ai()
