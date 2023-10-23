import numpy as np
import picos as pic


def random_state(n, n_points):
    real_current_matrix = np.random.rand(n, 1)
    real_current_matrix = np.matrix(real_current_matrix)

    imaginary_current_matrix = np.random.rand(n, 1)
    imaginary_current_matrix = np.matrix(imaginary_current_matrix)

    current_matrix = real_current_matrix + 1j * imaginary_current_matrix

    state = current_matrix * np.transpose(np.conj(current_matrix))
    return np.array(state / (n_points * state.trace()))


exchange = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

rho = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1 / 3, 1 / 3, 0.0, 1 / 3, 0.0, 0.0, 0.0],
                 [0.0, 1 / 3, 1 / 3, 0.0, 1 / 3, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 1 / 3, 1 / 3, 0.0, 1 / 3, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

white_noise = np.eye(8) / 8

def f_1(X_list, n_points):
    prob = pic.Problem()
    rho_list = list()
    for i in range(3):
        rho_list.append([])

    p = pic.RealVariable("p", 1)

    for i in range(n_points):
        rho_1 = pic.HermitianVariable('rho_1' + str(i), 2)
        rho_2 = pic.HermitianVariable('rho_2' + str(i), 2)
        rho_3 = pic.HermitianVariable('rho_3' + str(i), 2)

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
                rho_next += exchange * (X_list[index_1][index_2] @ rho_list[index_1][index_2]) * exchange
            else:
                rho_next += rho_list[index_1][index_2] @ X_list[index_1][index_2]

    prob.add_constraint(rho_next == (p * rho + ((1 - p) / 8) * np.eye(8)))
    prob.set_objective("max", p)
    prob.solve(solver="mosek", primals=True)

    if np.min([np.linalg.eigvals(current_rho) for current_rho in rho_list[0]]) < 0:
        print("ERROR")

    # print(rho_next)
    # print([np.linalg.eigvals(current_rho) for current_rho in rho_list[0]])
    # print(np.linalg.eigvals(rho_next))
    # print(p.value * rho + ((1 - p.value) / 8) * np.eye(8))
    # print(rho_next - (p.value * rho + ((1 - p.value) / 8) * np.eye(8)))
    # print(np.linalg.eigvals(rho_next - (p * rho + ((1 - p) / 8) * np.eye(8))))
    print("f_1", p.value)

    return np.array(rho_list), p.value


def f_2(X_list, n_points):
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

    rho_next = rho_list[0][0] @ X_list[0][0]
    for index_1 in range(3):
        for index_2 in range(n_points):
            if index_1 == 0 and index_2 == 0:
                continue
            if index_1 == 0:
                rho_next += rho_list[index_1][index_2] @ X_list[index_1][index_2]
            elif index_1 == 1:
                rho_next += exchange * (rho_list[index_1][index_2] @ X_list[index_1][index_2]) * exchange
            else:
                rho_next += X_list[index_1][index_2] @ rho_list[index_1][index_2]

    prob.add_constraint(rho_next == (p * rho + ((1 - p) / 8) * np.eye(8)))
    # prob.add_constraint((p * rho + ((1 - p) / 8) * np.eye(8)) - rho_next >> 0)
    # prob.add_constraint(p <= 1)
    # prob.add_constraint(pic.trace(rho_next) <= 1)
    prob.set_objective("max", p)
    # print(prob)
    # prob.maximize = p
    prob.solve(solver="mosek", primals=True)

    # print("rho_next", rho_next)
    if np.min([np.linalg.eigvals(current_rho) for current_rho in rho_list[0]]) < 0:
        print("ERROR")

    # print(np.linalg.eigvals(rho_next))
    # print(p.value * rho + ((1 - p.value) / 8) * np.eye(8))
    # print(rho_next - (p.value * rho + ((1 - p.value) / 8) * np.eye(8)))
    # print(np.linalg.eigvals(rho_next - (p * rho + ((1 - p) / 8) * np.eye(8))))
    print("f_2", p.value)
    print(rho_next.value)

    beta = (rho - white_noise).reshape(-1)
    beta_norm = np.linalg.norm(beta)

    alpha = (rho_next.value - white_noise).reshape(-1)
    cos_value = np.linalg.norm(beta.T * alpha) / (beta_norm * np.linalg.norm(alpha))
    distance_loss = 1000 * np.linalg.norm(alpha) * np.sqrt(1 - cos_value ** 2)
    scalar = np.linalg.norm(alpha) * cos_value / beta_norm

    return np.array(rho_list), p.value


if __name__ == "__main__":
    n_points = 300
    max_value = -1
    max_before_list = list()
    max_next_list = list()

    for i in range(300):
        X_list = [[random_state(2, n_points) for index_1 in range(n_points)] for index_2 in range(3)]
        length = 30
        current_before_list = X_list
        for j in range(length):
            X_list, p_value = f_2(X_list, n_points)
            # if p_value > max_value * 0.95:
            #     length = 50
            #
            # if p_value > max_value:
            #     max_value = p_value
            #     max_before_list = current_before_list
            #     max_next_list = X_list
            #     print("current max value:", max_value)

            X_list = [[X_list[index_1][index_2] * np.trace(np.matrix(current_before_list[index_1][index_2])) for index_2 in
                       range(n_points)] for index_1 in range(3)]
            current_before_list = X_list

            X_list, p_value = f_1(X_list, n_points)
            X_list = [[X_list[index_1][index_2] * np.trace(np.matrix(current_before_list[index_1][index_2])) for index_2 in
                       range(n_points)] for index_1 in range(3)]
            current_before_list = X_list

            # if p_value > max_value * 0.95:
            #     length = 50
            #
            # if p_value > max_value:
            #     max_value = p_value
            #     max_before_list = current_before_list
            #     max_next_list = X_list
            #     print("current max value:", max_value)
        print("Round", i)

    print("max_value", max_value)
    print(max_before_list)
    print(max_next_list)
