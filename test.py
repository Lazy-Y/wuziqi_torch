import torch

b = torch.tensor([list(range(4)) for i in range(3)])
print(b.shape)
b = b.view((-1, 3, 2, 2))
print(b)
# board = torch.reshape(b, (1, 3, 4))
# board = torch.transpose(board, 1, 2)
# print(board)
# kernel = torch.ones((3))

# bias = torch.tensor(0.5)
# out = torch.sum(board * kernel, axis=-1) + bias
# print(torch.reshape(out, (1, 2, 2)))

# print(torch.randn((5, 5)))
