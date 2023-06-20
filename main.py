import random

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

pauli_matrix_dict = {
    'I': np.matrix([[1, 0], [0, 1]]),
    'X': np.matrix([[0, 1], [1, 0]]),
    'Y': np.matrix([[0, -1j], [1j, 0]]),
    'Z': np.matrix([[1, 0], [0, -1]])
}

setting = 2


def generate_density_matrix(dim=4):
    s1 = list()
    s2 = list()
    for index in range(dim):
        temp_s1 = list()
        temp_s2 = list()
        for index2 in range(dim):
            temp_s1.append(random.uniform(-1, 1))
            temp_s2.append(random.uniform(-1, 1))
        s1.append(temp_s1)
        s2.append(temp_s2)
    s1 = np.matrix(s1)
    s2 = np.matrix(s2)
    s = s1 + 1j * s2
    s = np.matrix(s)  # Create a random Complex matrix H.
    s = s + s.H  # Generate GUE
    s = np.dot(s, s.H) / np.trace(np.dot(s, s.H))  # Make it positive, and normalize to unity.
    return s


def handle_density_matrix(density_matrix):
    density_matrix = np.matrix(density_matrix)
    result = list()
    for x in range(density_matrix.shape[0]):
        item = list()
        for y in range(density_matrix.shape[1]):
            if x > y:
                item.append(np.imag(density_matrix[x, y]))
            else:
                item.append(np.real(density_matrix[x, y]))
        result.append(item)
    return np.matrix(result)

def generate_density_matrix_feature(density_matrix):
    return [
        [density_matrix[0, 1], density_matrix[2, 3], density_matrix[0, 3] + density_matrix[1, 2], -(density_matrix[3, 0] + density_matrix[2, 1])],
        [density_matrix[1, 0], density_matrix[3, 2], density_matrix[3, 0] - density_matrix[2, 1], density_matrix[0, 3] - density_matrix[1, 2]],
        [(density_matrix[0, 0] - density_matrix[1, 1]) / 2, (density_matrix[2, 2] - density_matrix[3, 3]) / 2,
         density_matrix[0, 2] - density_matrix[1, 3], - density_matrix[2, 0] + density_matrix[3, 1]]
    ]


if __name__ == "__main__":
    # rho = generate_density_matrix()
    rho = [
        [0.5, 0, 0, 0.5],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0.5, 0, 0, 0.5]
    ]
    rho = handle_density_matrix(rho)
    print(rho)
    density_matrix_feature = generate_density_matrix_feature(rho)
    density_matrix_feature = torch.tensor(density_matrix_feature).double().to(device)

    phi_1 = torch.zeros(1, requires_grad=True)
    psi_1 = torch.zeros(1, requires_grad=True)

    phi_2 = torch.zeros(1, requires_grad=True)
    psi_2 = torch.zeros(1, requires_grad=True)

    feature_1 = torch.stack((torch.cos(phi_1) * torch.cos(psi_1), torch.cos(phi_1) * torch.sin(psi_1), torch.sin(phi_1))).double()
    feature_2 = torch.stack((torch.sin(phi_2) * torch.cos(psi_2), torch.sin(phi_2) * torch.sin(psi_2), torch.cos(phi_2))).double()

    settings_transport = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).double().to(device)
    settings_add = torch.transpose(torch.tensor([[1.0, 1.0, 0.0, 0.0]]).double().to(device), 0, 1)

    measure_matrix_1_0 = settings_add + torch.mm(settings_transport, feature_1)
    measure_matrix_1_1 = settings_add - torch.mm(settings_transport, feature_1)

    measure_matrix_2_0 = settings_add + torch.mm(settings_transport, feature_2)
    measure_matrix_2_1 = settings_add - torch.mm(settings_transport, feature_2)

    measurement_result_1_0 = torch.mm(density_matrix_feature, measure_matrix_1_0)
    measurement_result_1_1 = torch.mm(density_matrix_feature, measure_matrix_1_1)

    measurement_result_2_0 = torch.mm(density_matrix_feature, measure_matrix_2_0)
    measurement_result_2_1 = torch.mm(density_matrix_feature, measure_matrix_2_1)

    next_measurement_result_2_1 = 2 * measurement_result_1_0 - measurement_result_2_1
    next_measurement_result_1_1 = 2 * measurement_result_2_0 - measurement_result_1_1




