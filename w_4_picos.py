from w_3_picos_back import random_state
import numpy as np
import picos as pic

rho = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

exchange_1 = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

exchange_2 = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])


def f_1(X_list, n_points):
    prob = pic.Problem()
    rho_list = list()
    for i in range(3):
        rho_list.append([])

    p = pic.RealVariable("p", 1)

    for i in range(n_points):
        rho_1 = pic.SymmetricVariable('rho_1' + str(i), 4)
        rho_2 = pic.SymmetricVariable('rho_2' + str(i), 4)
        rho_3 = pic.SymmetricVariable('rho_3' + str(i), 4)

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
                rho_next += exchange_1 * (X_list[index_1][index_2] @ rho_list[index_1][index_2]) * exchange_1
            else:
                rho_next += exchange_2 * (X_list[index_1][index_2] @ rho_list[index_1][index_2]) * exchange_2

    prob.add_constraint(rho_next - (p * rho + ((1 - p) / 16) * np.eye(16)) >> 0)
    prob.add_constraint(p <= 1)
    prob.add_constraint(pic.trace(rho_next) <= 1)
    prob.set_objective("max", p)
    prob.solve(solver="mosek", primals=True)

    print("f_1", p.value)
    return np.array(rho_list), p.value


def f_2(X_list, n_points):
    prob = pic.Problem()
    rho_list = list()
    for i in range(3):
        rho_list.append([])

    p = pic.RealVariable("p", 1)

    for i in range(n_points):
        rho_1 = pic.SymmetricVariable('rho_1' + str(i), 4)
        rho_2 = pic.SymmetricVariable('rho_2' + str(i), 4)
        rho_3 = pic.SymmetricVariable('rho_3' + str(i), 4)

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
                rho_next += exchange_1 * (rho_list[index_1][index_2] @ X_list[index_1][index_2]) * exchange_1
            else:
                rho_next += exchange_2 * (rho_list[index_1][index_2] @ X_list[index_1][index_2]) * exchange_2

    prob.add_constraint(rho_next - (p * rho + ((1 - p) / 16) * np.eye(16)) >> 0)
    prob.add_constraint(p <= 1)
    prob.add_constraint(pic.trace(rho_next) <= 1)
    prob.set_objective("max", p)
    # prob.maximize = p
    prob.solve(solver="mosek", primals=True)

    print("f_2", p.value)

    return np.array(rho_list), p.value


if __name__ == "__main__":
    n_points = 100
    max_value = -1
    max_before_list = list()
    max_next_list = list()

    for i in range(300):
        X_list = [[random_state(4) for index_1 in range(n_points - 1)] + [np.eye(4)] for index_2 in range(3)]
        length = 12
        current_before_list = list()
        for j in range(length):
            # X_list = [
            #     [X_list[index_1][index_2] / np.matrix(X_list[index_1][index_2]).trace() for index_2 in range(n_points)]
            #     for index_1 in range(3)]
            X_list, p_value = f_1(X_list, n_points)

            if p_value > max_value:
                max_value = p_value
                max_before_list = current_before_list
                max_next_list = X_list
                print("current max value:", max_value)

            # X_list = [
            #     [X_list[index_1][index_2] / np.matrix(X_list[index_1][index_2]).trace() for index_2 in range(n_points)]
            #     for index_1 in range(3)]

            X_list, p_value = f_2(X_list, n_points)

            if p_value > max_value:
                max_value = p_value
                max_before_list = current_before_list
                max_next_list = X_list
                print("current max value:", max_value)

        print("Round", i)

    # print("max_value", max_value)
    # print(max_before_list)
    # print(max_next_list)
