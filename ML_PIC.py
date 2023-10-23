import numpy as np
import picos as pic
import torch
import torch.nn.functional as F

from partition_tools import generate_k_producible_partitions


def bubble_sort_steps(nums):
    steps = list()
    for i in range(len(nums)):
        for j in range(len(nums) - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                steps.append((j, j + 1))
    return steps


def swap_chars(s, i, j):
    lst = list(s)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)


class ML_PIC(object):
    def __init__(self, N, n_points, rho, partition_list, r):
        self.N = N
        self.n_points = n_points
        self.rho = rho
        self.r = r
        self.white_noise = torch.eye(2 ** N) / 2 ** N
        self.beta = torch.flatten(torch.tensor(rho, dtype=torch.complex128) - self.white_noise)
        self.beta_norm = torch.norm(self.beta)
        self.partition_list = partition_list

        self.exchange_matrix = list()
        for num1 in range(self.N):
            temp_matrix_list = list()
            for num2 in range(num1 + 1, self.N):
                the_matrix = np.zeros([2 ** self.N, 2 ** self.N])
                for number in range(2 ** self.N):
                    number_str = format(number, '0{}b'.format(self.N))
                    number_str = swap_chars(number_str, num1, num2)
                    number_23 = int(number_str, 2)
                    the_matrix[number, number_23] = 1
                temp_matrix_list.append(torch.tensor(the_matrix, dtype=torch.complex128))
            self.exchange_matrix.append(temp_matrix_list)

        self.exchange_list = list()
        for partition in self.partition_list:
            concatenated_list = [element for sublist in partition.partition_by_list for element in sublist]
            self.exchange_list.append(bubble_sort_steps(concatenated_list))

    def train(self, epoch):
        weights = torch.randn(self.n_points)
        opti_list = [weights]
        L_list = list()
        for index_point in range(self.n_points):
            point_index_list = list()
            for partition in self.partition_list:
                current_L = list()
                for part in partition.partition_by_list:
                    L_temp = torch.randn(2 ** len(part), 2 ** len(part), dtype=torch.complex128)
                    L_temp.requires_grad_(True)
                    current_L.append(L_temp)
                opti_list += current_L
                point_index_list.append(current_L)
            L_list.append(point_index_list)

        optimizer = torch.optim.SGD(opti_list, lr=0.01)

        for epoch in range(epoch):
            weights_normalized = F.normalize(torch.abs(weights), p=1, dim=0)
            target = torch.zeros(2 ** self.N, 2 ** self.N, dtype=torch.complex128)
            # L_1_torch_list = list()

            for j in range(self.n_points):

                for i in range(len(self.partition_list)):
                    current_L = L_list[j][i]



                L_1 = torch.matmul(L_1_list[j], L_1_list[j].conj().t())
                L_2 = torch.matmul(L_2_list[j], L_2_list[j].conj().t())
                L_1 = L_1 / L_1.trace()
                L_2 = L_2 / L_2.trace()

                # L_1_torch_list.append(L_1)

                if j % 3 == 0:
                    target += (weights_normalized[j] * torch.kron(L_1, L_2))
                elif j % 3 == 1:
                    target += (weights_normalized[j] * torch.matmul(torch.matmul(exchange_1, torch.kron(L_1, L_2)),
                                                                    exchange_1))
                else:
                    target += (weights_normalized[j] * torch.matmul(torch.matmul(exchange_2, torch.kron(L_1, L_2)),
                                                                    exchange_2))

            alpha = torch.flatten(target - self.white_noise)
            cos_value = torch.norm(torch.matmul(self.beta.T, alpha)) / (self.beta_norm * torch.norm(alpha))
            distance_loss = 1000 * torch.norm(alpha) * torch.sqrt(1 - cos_value ** 2)
            scalar = torch.norm(alpha) * cos_value / self.beta_norm

            target_loss = 1000 * torch.norm(target - torch.tensor(self.rho, dtype=torch.complex128))

            if distance_loss > self.r:
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

            # result = [[], [], []]
            # weights_result = weights_normalized.detach().numpy()
            # for j in range(self.n_points):
            #     result[j % 3].append(L_1_torch_list[j].detach().numpy() * weights_result[j])
            # L1_queue.append(result)

    # def sdp(self):
    #     prob = pic.Problem()
    #     rho_list = list()
    #     for i in range(3):
    #         rho_list.append([])
    #
    #     p = pic.RealVariable("p", 1)
    #
    #     for i in range(n_points):
    #         rho_1 = pic.HermitianVariable('rho_1' + str(i), 4)
    #         rho_2 = pic.HermitianVariable('rho_2' + str(i), 4)
    #         rho_3 = pic.HermitianVariable('rho_3' + str(i), 4)
    #
    #         rho_list[0].append(rho_1)
    #         rho_list[1].append(rho_2)
    #         rho_list[2].append(rho_3)
    #
    #         prob.add_constraint(rho_1 >> 0)
    #         prob.add_constraint(rho_2 >> 0)
    #         prob.add_constraint(rho_3 >> 0)
    #
    #     rho_next = X_list[0][0] @ rho_list[0][0]
    #     for index_1 in range(3):
    #         for index_2 in range(n_points):
    #             if index_1 == 0 and index_2 == 0:
    #                 continue
    #             if index_1 == 0:
    #                 rho_next += X_list[index_1][index_2] @ rho_list[index_1][index_2]
    #             elif index_1 == 1:
    #                 rho_next += exchange_1_np * (X_list[index_1][index_2] @ rho_list[index_1][index_2]) * exchange_1_np
    #             else:
    #                 rho_next += exchange_2_np * (X_list[index_1][index_2] @ rho_list[index_1][index_2]) * exchange_2_np
    #
    #     prob.add_constraint(rho_next == (p * rho + ((1 - p) / (2 ** N)) * np.eye(2 ** N)))
    #     prob.set_objective("max", p)
    #     prob.solve(solver="mosek", primals=True)
    #
    #     print("final", p.value)
    #     print(rho_next.value)
    #     return np.array(rho_list), p.value


if __name__ == "__main__":
    rho = np.zeros((32, 32))

    indices = [0, 15, 19, 28]  # Python uses 0-based indexing
    for index in indices:
        for index2 in indices:
            rho[index, index2] = 0.25

    partition_2_prod = generate_k_producible_partitions(5, 2)
    # print(bubble_sort_steps([element for sublist in partition_2_prod[0].partition_by_list for element in sublist]))
    current_class = ML_PIC(5, 100, rho, partition_2_prod, 1)
    current_class.train(1000)
