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
                steps.append([j, j + 1])
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
        self.exchange_matrix_np = list()
        for num1 in range(self.N):
            temp_matrix_list = list()
            temp_matrix_list_np = list()
            for num2 in range(num1 + 1, self.N):
                the_matrix = np.zeros([2 ** self.N, 2 ** self.N])
                for number in range(2 ** self.N):
                    number_str = format(number, '0{}b'.format(self.N))
                    number_str = swap_chars(number_str, num1, num2)
                    number_23 = int(number_str, 2)
                    the_matrix[number, number_23] = 1
                temp_matrix_list.append(torch.tensor(the_matrix, dtype=torch.complex128))
                temp_matrix_list_np.append(np.matrix(the_matrix))
            self.exchange_matrix.append(temp_matrix_list)
            self.exchange_matrix_np.append(temp_matrix_list_np)

        self.exchange_list = list()
        self.partition_max_part_list = list() # todo
        for partition in self.partition_list:
            concatenated_list = [element for sublist in partition.partition_by_list for element in sublist]
            self.exchange_list.append(bubble_sort_steps(concatenated_list))


    def train(self, epoch):
        weights = torch.randn(self.n_points * len(self.partition_list))
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

            for j in range(self.n_points):
                for i in range(len(self.partition_list)):
                    current_L = L_list[j][i]

                    L_0 = torch.matmul(current_L[0], current_L[0].conj().t())
                    L_0 = L_0 / L_0.trace()
                    L_1 = torch.matmul(current_L[1], current_L[1].conj().t())
                    L_1 = L_1 / L_1.trace()

                    current_target = torch.kron(L_0, L_1)
                    for L_index in range(2, len(current_L)):
                        L = torch.matmul(current_L[L_index], current_L[L_index].conj().t())
                        L = L / L.trace()
                        current_target = torch.kron(current_target, L)

                    for exchange_pair in self.exchange_list[i]:
                        current_target = torch.matmul(torch.matmul(self.exchange_matrix[exchange_pair[0]][exchange_pair[1] - exchange_pair[0] - 1], current_target),self.exchange_matrix[exchange_pair[0]][exchange_pair[1] - exchange_pair[0] - 1])

                    current_target = weights_normalized[j * len(self.partition_list) + i] * current_target
                    target += current_target

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

            result = list()
            weights_result = weights_normalized.detach().numpy()
            for j in range(self.n_points):
                temp_point_list = list()
                for i in range(len(self.partition_list)):
                    t = weights_normalized[j * len(self.partition_list) + i]
                    current_L = list()

            # for j in range(self.n_points):
            #     result[j % 3].append(L_1_torch_list[j].detach().numpy() * weights_result[j])
            # L1_queue.append(result)

    def sdp(self):
        prob = pic.Problem()
        rho_list = list()
        for i in range(len(self.partition_list)):
            rho_list.append([])

        p = pic.RealVariable("p", 1)

        for i in range(self.n_points):
            for j in range(len(self.partition_list)):
                sdp_rho = pic.HermitianVariable('rho_' + str(i) + '_' + str(j), 4)
                rho_list[i].append(sdp_rho)
                prob.add_constraint(sdp_rho >> 0)

        rho_next = X_list[0][0] @ rho_list[0][0]
        for index_1 in range(self.n_points):
            for index_2 in range(len(self.partition_list)):
                if index_1 == 0 and index_2 == 0:
                    continue
                if index_1 == 0:
                    rho_next += X_list[index_1][index_2] @ rho_list[index_1][index_2]
                elif index_1 == 1:
                    rho_next += exchange_1_np * (X_list[index_1][index_2] @ rho_list[index_1][index_2]) * exchange_1_np
                else:
                    rho_next += exchange_2_np * (X_list[index_1][index_2] @ rho_list[index_1][index_2]) * exchange_2_np

        prob.add_constraint(rho_next == (p * rho + ((1 - p) / (2 ** self.N)) * np.eye(2 ** self.N)))
        prob.set_objective("max", p)
        prob.solve(solver="mosek", primals=True)

        print("final", p.value)
        print(rho_next.value)
        return np.array(rho_list), p.value


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
    current_class.sdp()
