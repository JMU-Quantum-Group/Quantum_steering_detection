import torch
import torch.nn.functional as F
import picos as pic
import numpy as np

from collections import deque

n = 4
n_points = 999

N = 4

L_1_list = list()
L_2_list = list()

rho = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.25, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.25, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.25, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.25, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

exchange_1 = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
exchange_1 = torch.tensor(exchange_1, dtype=torch.complex128)

exchange_1_np = np.matrix(exchange_1)

exchange_2 = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
exchange_2 = torch.tensor(exchange_2, dtype=torch.complex128)

exchange_2_np = np.matrix(exchange_2)

white_noise = torch.eye(2 ** N) / 2 ** N

beta = torch.flatten(torch.tensor(rho, dtype=torch.complex128) - white_noise)
beta_norm = torch.norm(beta)

for i in range(n_points):
    L_1 = torch.randn(n, n, dtype=torch.complex128)
    L_1.requires_grad_(True)
    L_1_list.append(L_1)

    L_2 = torch.randn(n, n, dtype=torch.complex128)
    L_2.requires_grad_(True)
    L_2_list.append(L_2)

weights = torch.randn(n_points)

opti_list = L_1_list + L_2_list + [weights]

# 设置优化器
optimizer = torch.optim.SGD(opti_list, lr=0.01)

L1_queue = deque(maxlen=5)

for epoch in range(20000):
    weights_normalized = F.normalize(torch.abs(weights), p=1, dim=0)
    # 计算半正定矩阵
    target = torch.zeros(2 ** N, 2 ** N, dtype=torch.complex128)
    L_1_torch_list = list()
    for j in range(n_points):
        L_1 = torch.matmul(L_1_list[j], L_1_list[j].conj().t())
        L_2 = torch.matmul(L_2_list[j], L_2_list[j].conj().t())
        L_1 = L_1 / L_1.trace()
        L_2 = L_2 / L_2.trace()

        L_1_torch_list.append(L_1)

        if j % 3 == 0:
            target += (weights_normalized[j] * torch.kron(L_1, L_2))
        elif j % 3 == 1:
            target += (weights_normalized[j] * torch.matmul(torch.matmul(exchange_1, torch.kron(L_1, L_2)), exchange_1))
        else:
            target += (weights_normalized[j] * torch.matmul(torch.matmul(exchange_2, torch.kron(L_1, L_2)), exchange_2))

    alpha = torch.flatten(target - white_noise)
    cos_value = torch.norm(torch.matmul(beta.T, alpha)) / (beta_norm * torch.norm(alpha))
    distance_loss = 1000 * torch.norm(alpha) * torch.sqrt(1 - cos_value ** 2)
    scalar = torch.norm(alpha) * cos_value / beta_norm

    target_loss = 1000 * torch.norm(target - torch.tensor(rho, dtype=torch.complex128))

    if distance_loss > 1:
        optimizer.zero_grad()
        distance_loss.backward()
        optimizer.step()
        print(
            f'Epoch {epoch} distance: Scalar = {scalar.item()}, Scalar Loss = {target_loss.item()}, Distance Loss = {distance_loss.item()}')
    else:
        optimizer.zero_grad()
        target_loss.backward()
        optimizer.step()
        print(
            f'Epoch {epoch} scalar: Scalar = {scalar.item()}, Scalar Loss = {target_loss.item()}, Distance Loss = {distance_loss.item()}')

    result = [[], [], []]
    weights_result = weights_normalized.detach().numpy()
    for j in range(n_points):
        result[j % 3].append(L_1_torch_list[j].detach().numpy() * weights_result[j])
    L1_queue.append(result)


def f_1(X_list, n_points):
    prob = pic.Problem()
    rho_list = list()
    for i in range(3):
        rho_list.append([])

    p = pic.RealVariable("p", 1)

    for i in range(n_points):
        rho_1 = pic.HermitianVariable('rho_1' + str(i), 4)
        rho_2 = pic.HermitianVariable('rho_2' + str(i), 4)
        rho_3 = pic.HermitianVariable('rho_3' + str(i), 4)

        rho_list[0].append(rho_1)
        rho_list[1].append(rho_2)
        rho_list[2].append(rho_3)

        prob.add_constraint(rho_1 >> 0)
        prob.add_constraint(rho_2 >> 0)
        prob.add_constraint(rho_3 >> 0)

    rho_next = X_list[0][0] @ rho_list[0][0]
    for index_1 in range(3):
        for index_2 in range(n_points):
            if index_1 == 0 and index_2 == 0:
                continue
            if index_1 == 0:
                rho_next += X_list[index_1][index_2] @ rho_list[index_1][index_2]
            elif index_1 == 1:
                rho_next += exchange_1_np * (X_list[index_1][index_2] @ rho_list[index_1][index_2]) * exchange_1_np
            else:
                rho_next += exchange_2_np * (X_list[index_1][index_2] @ rho_list[index_1][index_2]) * exchange_2_np

    prob.add_constraint(rho_next == (p * rho + ((1 - p) / (2 ** N)) * np.eye(2 ** N)))
    prob.set_objective("max", p)
    prob.solve(solver="mosek", primals=True)

    print("final", p.value)
    print(rho_next.value)
    return np.array(rho_list), p.value


X_list, p_value = f_1(L1_queue[0], 333)
