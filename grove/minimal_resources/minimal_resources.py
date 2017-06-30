"""
Minimal resources as per arXiv/1701.08213

"""

from pyquil.paulis import *


def generator_matrix(pauli_sum):
    """

    :param PauliSum pauli_sum:
    :return:
    """
    qubits = pauli_sum.get_qubits()

    n = len(qubits)

    mappings = {qubits[i]: i for i in xrange(len(qubits))}
    Gx_transposed = np.empty((0, n), int)
    Gz_transposed = np.empty((0, n), int)

    for pt in pauli_sum:
        row_x = np.zeros(shape=(1, n), dtype=int)
        row_z = np.zeros(shape=(1, n), dtype=int)
        for qubit, term in list(pt):
            if term == 'I':
                continue
            if term in {'X', 'Y'}:
                row_x[0][mappings[qubit]] = 1
            if term in {'Z', 'Y'}:
                row_z[0][mappings[qubit]] = 1

        Gx_transposed = np.append(Gx_transposed, row_x, axis=0)
        Gz_transposed = np.append(Gz_transposed, row_z, axis=0)

    return Gx_transposed.transpose(), Gz_transposed.transpose()


def parity_check_matrix(pauli_sum):
    """
    Finds parity check matrix out of a pauli sum

    :param pauli_sum:
    :return:
    """
    Gx, Gz = generator_matrix(pauli_sum)
    E = np.append(Gz.transpose(), Gx.transpose(), axis=1)

    return E


def make_binary_reduced_row_echelon(matrix):
    """
    Makes matrix reduced row echelon

    :param matrix:
    :return:
    """
    matrix_copy = np.array(matrix)
    m, n = matrix_copy.shape
    for k in xrange(min(m, n)):
        i_max = k - 1
        for row in xrange(k, m):
            if matrix_copy[row, k] == 1:
                i_max = row
                break
        if i_max == k - 1:
            continue

        matrix_copy[[k, i_max]] = matrix_copy[[i_max, k]]

        for i in xrange(k + 1, m):
            if matrix_copy[i, k] == 0:
                continue
            for j in xrange(k, n):
                matrix_copy[i, j] = matrix_copy[i, j] ^ matrix_copy[k, j]

    for row in xrange(min(m - 1, n - 1), -1, -1):
        col_max = 0
        for col in xrange(n):
            if matrix_copy[row, col] == 1:
                break
            col_max += 1
        if col_max == n:
            continue
        for above_row in xrange(row - 1, -1, -1):
            if matrix_copy[above_row, col_max] == 0:
                continue
            for j in xrange(col_max, n):
                matrix_copy[above_row, j] = matrix_copy[above_row, j] ^ matrix_copy[row, j]

    return matrix_copy

if __name__ == '__main__':
    no_coef_hydrogen = sZ(1) + sZ(2) + sZ(3) + sZ(4) + \
                       sZ(1) * sZ(2) + sZ(1) * sZ(3) + sZ(1) * sZ(4) + \
                       sZ(2) * sZ(3) + sZ(2) * sZ(4) + sZ(3) * sZ(4) + \
                       sY(1) * sY(2) * sX(3) * sX(4) + \
                       sX(1) * sY(2) * sY(3) * sX(4) + \
                       sY(1) * sX(2) * sX(3) * sY(4) + \
                       sX(1) * sX(2) * sY(3) * sY(4)

    Gx, Gz = generator_matrix(no_coef_hydrogen)
    print "Gx\n", Gx
    print "Gz\n", Gz
    print '---------------------------------------------------------'
    E = parity_check_matrix(no_coef_hydrogen)
    print "E\n", E
    print '---------------------------------------------------------'

    rre_E = make_binary_reduced_row_echelon(E)
    rre_E_nontrivial = rre_E[~np.all(rre_E == 0, axis=1)]
    print "E_tilde\n", rre_E_nontrivial
