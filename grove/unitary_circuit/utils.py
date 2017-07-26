"""Utils for generating circuits from unitaries."""
import numpy as np
import pyquil.quil as pq
from pyquil.gates import *
from scipy.linalg import sqrtm


### Controlled Single Qubit Gates ###
############################################################################
# see  http://d.umn.edu/~vvanchur/2015PHYS4071/Chapter4.pdf
# http://fab.cba.mit.edu/classes/862.16/notes/computation/Barenco-1995.pdf
###########################################################################
def one_qubit_controlled(U, control, target):
    """
    Get a controlled version of an arbitrary single qubit gate

    :param ndarray U: a 2x2 unitary matrix
    :param int control: the single control qubit
    :param int target: the single target qubit
    :return: The program representing applying a controlled-U gate with
             control control and target target.
    :rtype: Program
    """
    if len(U.shape) != 2:
        raise ValueError("U must be a 2 dimensional matrix")
    if not U.shape[0] == U.shape[1] == 2:
        raise ValueError("U must be a 2x2 matrix")
    if not np.allclose(np.eye(2), U.conj().T * U):
        raise ValueError("U must be unitary")
    if control < 0:
        raise ValueError("Control qubit must be nonnegative")
    if target < 0:
        raise ValueError("Target qubit must be nonnegative")
    if control == target:
        raise ValueError("Control and target qubits must be distinct")

    params = _one_qubit_gate_params(U)
    return _one_qubit_controlled_from_unitary_params(params, control, target)


def _one_qubit_gate_params(U):
    """
    Decompose U into e^i*alpha * RZ(beta)*RY(gamma)*RZ(delta).
    :param U: a 2x2 unitary matrix
    :return: alpha, beta, gamma, delta
    :rtype: tuple
    """
    # +0j to ensure the complex square root is taken
    d = np.sqrt(np.linalg.det(U) + 0j)
    U = U / d
    alpha = np.angle(d)
    if U[0, 0] == 0:
        beta = np.angle(U[1, 0])
        delta = -beta
    elif U[1, 0] == 0:
        beta = -np.angle(U[0, 0])
        delta = beta
    else:
        beta = -np.angle(U[0, 0]) + np.angle(U[1, 0])
        delta = -np.angle(U[0, 0]) - np.angle(U[1, 0])

    gamma = 2 * np.arctan2(np.abs(U[1, 0]), np.abs(U[0, 0]))
    return alpha, beta, gamma, delta


def _one_qubit_controlled_from_unitary_params(params, control, target):
    """
    Get the controlled version of the unitary, given angular parameters.
    Uses PHASE, RZ, RY and CNOT gates.

    :param params: tuple in the form (alpha, beta, gamma, delta)
    :param control: the control qubit
    :param target: the target qubit
    :return: the program that simulates acting a controlled
             unitary on the target, given the control.
    :rtype: Program
    """
    p = pq.Program()
    alpha, beta, gamma, delta = params

    # C
    if delta != beta:
        p.inst(RZ((delta - beta) / 2, target))

    # CNOT
    p.inst(CNOT(control, target))

    # B
    if delta != -beta:
        p.inst(RZ(-(delta + beta) / 2, target))
    if gamma != 0:
        p.inst(RY(-gamma / 2, target))

    # CNOT
    p.inst(CNOT(control, target))

    # A
    if gamma != 0:
        p.inst(RY(gamma / 2, target))
    if beta != 0:
        p.inst(RZ(beta, target))

    # PHASE SHIFT
    if alpha != 0:
        p.inst(PHASE(alpha, control))

    return p


def _one_qubit_gate_from_unitary_params(params, target):
    p = pq.Program()
    alpha, beta, gamma, delta = params

    # C
    if delta != beta:
        p.inst(RZ((delta - beta) / 2, target))

    # X
    p.inst(X(target))

    # B
    if delta != -beta:
        p.inst(RZ(-(delta + beta) / 2, target))
    if gamma != 0:
        p.inst(RY(-gamma / 2, target))

    # X
    p.inst(X(target))

    # A
    if gamma != 0:
        p.inst(RY(gamma / 2, target))
    if beta != 0:
        p.inst(RZ(beta, target))

    # PHASE
    if alpha != 0:
        p.inst(RZ(-2*alpha, target))
        p.inst(PHASE(2*alpha, target))

    return p

# X = e^i*pi/2*RZ(-pi/2)*RY(pi)*RZ(pi/2)
#   = PHASE(pi/2)*RZ(-pi/2)*RZ(-pi/2)*RY(pi)*RZ(pi/2)
# PHASE(alpha) = e^(-i*pi/2) * RZ(alpha)
def n_qubit_controlled_RZ(controls, target, theta):
    """
        :param controls: The list of control qubits
        :param target: The target qubit
        :param theta: The angle of rotation
        :return: the program that applies a RZ(theta) gate to target, given controls
        :rtype: Program
    """
    if len(controls) == 0:
        return pq.Program().inst(RZ(theta, target))
    p = pq.Program()
    p += _one_qubit_controlled_from_unitary_params(
        (0, theta / 4, theta / 4, 0), controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += _one_qubit_controlled_from_unitary_params(
        (0, -theta / 4, -theta / 4, 0), controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += n_qubit_controlled_RZ(controls[:-1], target, theta / 2)
    return p


def n_qubit_controlled_RY(controls, target, theta):
    """
    :param controls: The list of control qubits
    :param target: The target qubit
    :param theta: The angle of rotation
    :return: the program that applies a RY(theta) gate to target, given controls
    :rtype: Program
    """
    if len(controls) == 0:
        return pq.Program().inst(RY(theta, target))
    p = pq.Program()
    p += _one_qubit_controlled_from_unitary_params((0, 0, 0, theta / 2),
                                                      controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += _one_qubit_controlled_from_unitary_params((0, 0, 0, -theta / 2),
                                                      controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += n_qubit_controlled_RY(controls[:-1], target, theta / 2)
    return p


def n_qubit_controlled_PHASE(controls, target, theta):
    """
    :param controls: The list of control qubits
    :param target: The target qubit
    :param theta: The angle of rotation
    :return: the program that applies a PHASE(theta) gate to target, given controls
    :rtype: Program
    """
    if len(controls) == 0:
        return pq.Program().inst(PHASE(theta, target))
    p = pq.Program()
    p += _one_qubit_controlled_from_unitary_params(
        (theta / 4, theta / 4, theta / 4, 0), controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += _one_qubit_controlled_from_unitary_params(
        (-theta / 4, -theta / 4, -theta / 4, 0), controls[-1], target)
    p += n_qubit_controlled_X(controls[:-1], controls[-1])
    p += n_qubit_controlled_PHASE(controls[:-1], target, theta / 2)
    return p


def n_qubit_controlled_X(controls, target, scratch_bit=None):
    """
    :param controls: The list of control qubits
    :param target: The target qubit
    :return: the program that applies a X gate to target, given controls (i.e. n-1 Toffoli)
    :rtype: Program
    """
    n_controls = len(controls)
    if n_controls == 0:
        return pq.Program().inst(X(target))
    if n_controls == 1:
        return pq.Program().inst(CNOT(controls[0], target))
    if n_controls == 2:
        return pq.Program().inst(CCNOT(controls[0], controls[1], target))

    p = pq.Program()
    free_bit = scratch_bit is None
    if free_bit:
        scratch_bit = p.alloc()

    n = n_controls + 2
    m = int(n / 2)

    p += n_qubit_controlled_X(controls[-m:], scratch_bit, target)
    p += n_qubit_controlled_X(controls[:-m] + [scratch_bit], target,
                              controls[-1])
    p += n_qubit_controlled_X(controls[-m:], scratch_bit, target)
    p += n_qubit_controlled_X(controls[:-m] + [scratch_bit], target,
                              controls[-1])

    if free_bit:
        p.free(scratch_bit)

    return p


def n_qubit_control(controls, target, u):
    """
    Returns a controlled u gate with n-1 controls.

    Does not define new gates. Follows arXiv:quant-ph/9503016. Uses the same format as in grove.grover.grover.

    :param controls: The indices of the qubits to condition the gate on.
    :param target: The index of the target of the gate.
    :param u: The unitary gate to be controlled, given as a numpy array.
    :return: The controlled gate.
    :rtype: Program
    """

    def controlled_program_builder(controls, target, target_gate):

        p = pq.Program()

        params = _one_qubit_gate_params(target_gate)

        sqrt_gate = sqrtm(target_gate)
        sqrt_params = _one_qubit_gate_params(sqrt_gate)

        adj_sqrt_params = _one_qubit_gate_params(np.conj(sqrt_gate).T)

        if len(controls) == 0:
            p += _one_qubit_gate_from_unitary_params(params, target)

        elif len(controls) == 1:
            # controlled U
            p += _one_qubit_controlled_from_unitary_params(params,
                                                              controls[0],
                                                              target)

        else:
            # controlled V
            many_toff = n_qubit_controlled_X(controls[:-1], controls[-1])

            p += _one_qubit_controlled_from_unitary_params(sqrt_params,
                                                              controls[-1],
                                                              target)

            p += many_toff

            # controlled V_adj
            p += _one_qubit_controlled_from_unitary_params(adj_sqrt_params,
                                                              controls[-1],
                                                              target)

            p += many_toff

            # n-2 controlled V
            p += controlled_program_builder(controls[:-1], target, sqrt_gate)

        return p

    p = controlled_program_builder(controls, target, u)
    return p
