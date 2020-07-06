import numpy as np
import snoop
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsnooper
torchsnooper.register_snoop()
snoop.install(enabled=False)

BOARD_SIZE = 5

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


class AI(nn.Module):
    def __init__(self):
        super(AI, self).__init__()
        self.conv1_num_channels = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2,
                      out_channels=self.conv1_num_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.conv1_num_channels)
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
        self.reset()

    def reset(self):
        self.optimizer = optim.Adam(self.parameters())

    @snoop()
    def forward(self, board):
        """
        Args:
            board ([ndarray / tensor shape (-1, BS, BS)])
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
            board_tensor == 2, board_ones, board_zeros)
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


def test_input():
    board_np = np.arange(BOARD_SIZE ** 2) % 3
    board_np = np.reshape(board_np, (1, BOARD_SIZE, BOARD_SIZE))
    print(board_np)
    ai = AI()
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
    board_np[2, 2] = 2
    print(board_np)
    action = coord_to_action(1, 1)
    ai = AI()
    prev = ai(np.expand_dims(board_np, axis=0))
    prev = torch.reshape(prev, (1, BOARD_SIZE, BOARD_SIZE))
    for _ in range(200):
        ai.train(action, board_np)
    curr = ai(np.expand_dims(board_np, axis=0))
    curr = torch.reshape(curr, (1, BOARD_SIZE, BOARD_SIZE))
    print(prev)
    print(curr)
    print(curr - prev)


if __name__ == "__main__":
    # test_input()
    test_train()
    # test_util()
