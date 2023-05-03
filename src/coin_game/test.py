import torch 
a = torch.Tensor([[1, 1.1, 1.2, 1.3], [2, 2.1, 2.2, 2.3], [3, 3.1, 3.2,3.3]])
b = torch.Tensor([[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
c = torch.stack([a, b], -1)
# print(a.shape)
# print(c, c.shape)

# d = c.reshape(12, -1)
# print(d)
# print(d[:, 0], d[:,1])

# print(c[:, :, 0] + c[:, :, 1])
# print(c[:, 0], c[:, 1])
# print(torch.stack([c[:, 0], c[:, 1]], 1))


# a = torch.Tensor([1, 1.1, 1.2])
# b = torch.Tensor([4, 4, 4])
# c = torch.stack([a, b], -1)
# print(c, c.shape)
# print(c[:, 0], c[:, 1])

grid_size = 3
batch_size = 3
n = 4

red_pos_flat = torch.randint(grid_size * grid_size, size=(batch_size * n,))
print(red_pos_flat)
red_pos_flat = red_pos_flat.reshape(batch_size, n)

red_pos = torch.stack((red_pos_flat // grid_size, red_pos_flat % grid_size), dim=-1)
print(red_pos, red_pos.shape)

red_pos_flat = red_pos.reshape(batch_size * n, -1)[ :, 0] * grid_size + red_pos.reshape(batch_size *n, -1)[:, 1]
print(red_pos_flat.shape)
red_pos_flat = red_pos_flat.reshape(batch_size* n)
print(red_pos_flat)

state = torch.zeros((batch_size*n, 4, grid_size * grid_size))
# print("state", state, state.shape)
print(state[:, 0].shape)
print()
print(red_pos_flat[:, None])

a = torch.Tensor([[1,2,3], [4,5,6]])
print(a[:, None], a[:, None].shape)
b = torch.Tensor([1,2,3])
print(b[:, None], b[:, None].shape)

print(red_pos_flat[:, None], red_pos_flat[:, None].shape)
state[:, 0].scatter_(1, red_pos_flat[:,  None], 1)
print("state[:, 0]", state[:, 0])

red_coin_pos_flat = torch.Tensor([1,2,3])
red_coin_pos_flat.expand(3, n)
print(red_coin_pos_flat)
print(a.detach())