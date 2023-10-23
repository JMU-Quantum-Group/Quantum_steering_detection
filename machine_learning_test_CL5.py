import torch
import torch.nn.functional as F
import picos as pic
import numpy as np

# 矩阵大小
N = 5
n_points = 1000

L_1_list = list()
L_2_list = list()
L_3_list = list()
L_4_list = list()

p = 0.06
# p = 7 / 23

rho = np.zeros((32, 32))

indices = [0, 15, 19, 28]  # Python uses 0-based indexing
for index in indices:
    for index2 in indices:
        rho[index, index2] = 0.25

np.set_printoptions(threshold=np.inf)


def swap_chars(s, i, j):
    lst = list(s)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)


def exchange_matrix(num1, num2):
    the_matrix = np.zeros([2 ** N, 2 ** N])
    for number in range(2 ** N):
        number_str = format(number, '0{}b'.format(N))
        number_str = swap_chars(number_str, num1, num2)
        number_23 = int(number_str, 2)
        the_matrix[number, number_23] = 1
    return the_matrix


exchange_12 = torch.tensor(exchange_matrix(0, 1), dtype=torch.cfloat)
exchange_13 = torch.tensor(exchange_matrix(0, 2), dtype=torch.cfloat)
exchange_14 = torch.tensor(exchange_matrix(0, 3), dtype=torch.cfloat)
exchange_15 = torch.tensor(exchange_matrix(0, 4), dtype=torch.cfloat)
exchange_23 = torch.tensor(exchange_matrix(1, 2), dtype=torch.cfloat)
exchange_24 = torch.tensor(exchange_matrix(1, 3), dtype=torch.cfloat)
exchange_25 = torch.tensor(exchange_matrix(1, 4), dtype=torch.cfloat)
exchange_34 = torch.tensor(exchange_matrix(2, 3), dtype=torch.cfloat)
exchange_35 = torch.tensor(exchange_matrix(2, 4), dtype=torch.cfloat)
exchange_45 = torch.tensor(exchange_matrix(3, 4), dtype=torch.cfloat)

matrix = p * torch.tensor(rho, dtype=torch.cfloat) + ((1 - p) / 32) * torch.eye(32)

loss_weight = list()
zero_loss_weight = list()
for line in matrix:
    current_line = list()
    zero_current_line = list()
    for item in line:
        if item != 0.0:
            current_line.append(1000)
            zero_current_line.append(1)
        else:
            current_line.append(1)
            zero_current_line.append(1000)
    loss_weight.append(current_line)
    zero_loss_weight.append(zero_current_line)

for i in range(n_points):
    L_1 = torch.randn(4, 4, dtype=torch.cfloat)
    L_1.requires_grad_(True)
    L_1_list.append(L_1)

    L_2 = torch.randn(2, 2, dtype=torch.cfloat)
    L_2.requires_grad_(True)
    L_2_list.append(L_2)

    L_3 = torch.randn(2, 2, dtype=torch.cfloat)
    L_3.requires_grad_(True)
    L_3_list.append(L_3)

    L_4 = torch.randn(2, 2, dtype=torch.cfloat)
    L_4.requires_grad_(True)
    L_4_list.append(L_4)

weights = torch.randn(n_points)

opti_list = L_1_list + L_2_list + L_3_list + L_4_list + [weights]

# 设置优化器
optimizer = torch.optim.SGD(opti_list, lr=0.01)

for epoch in range(150000):
    weights_normalized = F.normalize(torch.abs(weights), p=1, dim=0)
    # 计算半正定矩阵
    target = torch.zeros(2 ** N, 2 ** N, dtype=torch.cfloat)
    for j in range(n_points):
        L_1 = torch.matmul(L_1_list[j], L_1_list[j].conj().t())
        L_2 = torch.matmul(L_2_list[j], L_2_list[j].conj().t())
        L_3 = torch.matmul(L_3_list[j], L_3_list[j].conj().t())
        L_4 = torch.matmul(L_4_list[j], L_4_list[j].conj().t())

        L_1 = L_1 / L_1.trace()
        L_2 = L_2 / L_2.trace()
        L_3 = L_3 / L_3.trace()
        L_4 = L_4 / L_4.trace()

        if j % 3 == 0:
            target += (weights_normalized[j] * torch.kron(torch.kron(torch.kron(L_1, L_2), L_3), L_4))
        elif j % 3 == 1:
            target += (weights_normalized[j] * torch.matmul(
                torch.matmul(exchange_23, torch.kron(torch.kron(torch.kron(L_1, L_2), L_3), L_4)), exchange_23))
        elif j % 3 == 2:
            target += (weights_normalized[j] * torch.matmul(
                torch.matmul(exchange_24, torch.kron(torch.kron(torch.kron(L_1, L_2), L_3), L_4)), exchange_24))
        elif j % 3 == 3:
            target += (weights_normalized[j] * torch.matmul(
                torch.matmul(exchange_25, torch.kron(torch.kron(torch.kron(L_1, L_2), L_3), L_4)), exchange_25))
        elif j % 3 == 4:
            target += (weights_normalized[j] * torch.kron(torch.kron(torch.kron(L_2, L_1), L_3), L_4))
        elif j % 3 == 5:
            target += (weights_normalized[j] * torch.matmul(
                torch.matmul(exchange_34, torch.kron(torch.kron(torch.kron(L_2, L_1), L_3), L_4)), exchange_34))
        elif j % 3 == 6:
            target += (weights_normalized[j] * torch.matmul(
                torch.matmul(exchange_35, torch.kron(torch.kron(torch.kron(L_2, L_1), L_3), L_4)), exchange_35))
        elif j % 3 == 7:
            target += (weights_normalized[j] * torch.kron(torch.kron(torch.kron(L_2, L_3), L_1), L_4))
        elif j % 3 == 8:
            target += (weights_normalized[j] * torch.matmul(
                torch.matmul(exchange_45, torch.kron(torch.kron(torch.kron(L_2, L_3), L_1), L_4)), exchange_45))
        else:
            target += (weights_normalized[j] * torch.kron(torch.kron(torch.kron(L_2, L_3), L_4), L_1))

    loss = torch.abs(torch.flatten(matrix - target)).sum()
    train_loss = torch.dot(torch.abs(torch.flatten(matrix - target)),
                           torch.flatten(torch.tensor(loss_weight, dtype=torch.float))).sum()
    zero_loss = torch.dot(torch.abs(torch.flatten(matrix - target)),
                          torch.flatten(torch.tensor(zero_loss_weight, dtype=torch.float))).sum()

    max_loss = torch.abs(torch.flatten(matrix - target)).max()

    if train_loss.item() * 22 > zero_loss.item():
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print(
            f'Epoch {epoch}: Loss = {loss.item()}, Train Loss = {train_loss.item()}, Zero Loss = {zero_loss.item()}, Max Loss = {max_loss.item()}')
    else:
        optimizer.zero_grad()
        zero_loss.backward()
        optimizer.step()
        print(
            f'Epoch {epoch}: Loss = {loss.item()}, Train Loss = {train_loss.item()}, Zero Loss = {zero_loss.item()}, Max Loss = {max_loss.item()}')

print(f'{target.detach().numpy()}')
print(f'{target.detach().numpy().trace()}')
