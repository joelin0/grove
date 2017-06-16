"""Utils for Black Box Algorithm."""

import pyquil.quil as pq
from pyquil.gates import *
import numpy as np
from pyquil.gates import CCNOT
from grove.unitary_circuit.utils import *
from pyquil.forest import Connection

def try_and_or():
    p = pq.Program()
    q1 = p.alloc()
    q2 = p.alloc()
    p.inst(X(q1))
    sb = add_or(p, q1, q2)
    print p.out()

    cxn = Connection()
    result = cxn.run_and_measure(p, [q.index() for q in [q1, q2, sb]])
    print result

def test_two_bit_to_one_bit():
    p = pq.Program()
    q1 = p.alloc()
    q2 = p.alloc()
    p.inst(X(q1))

    f = [1, 1, 1, 1]
    num_range_bits = 1
    outbits = add_function(p, lambda x: f[x], [q2, q1], num_range_bits)
    print p.out()
    print len(p.get_qubits())

    cxn = Connection()
    result = cxn.run_and_measure(p, [q.index() for q in outbits])
    print result

def test_three_to_three():
    """
    Currently doesn't run cuz too many qubits
    :return:
    """
    p = pq.Program()
    q1 = p.alloc()
    q2 = p.alloc()
    q3 = p.alloc()
    p.inst(X(q2))
    p.inst(X(q1))

    f = [0, 1, 2, 3, 4, 5, 6, 7]
    num_range_bits = 3
    outbits = add_function(p, lambda x: f[x], [q3, q2, q1], num_range_bits)
    print p.out()
    print len(p.get_qubits())

    # cxn = Connection()
    # result = cxn.run_and_measure(p, [q.index() for q in outbits])
    # print result

def test_and_all():
    p = pq.Program()
    q1 = p.alloc()
    q2 = p.alloc()
    q3 = p.alloc()
    p.inst(X(q3))
    p.inst(X(q2))
    p.inst(X(q1))

    p, sb = and_all(p, [q3, q2, q1])

    print p.out()

    cxn = Connection()
    result = cxn.run_and_measure(p, [sb.index()])

    print p.get_qubits()
    print result

def test_or_all():
    p = pq.Program()
    q1 = p.alloc()
    q2 = p.alloc()
    q3 = p.alloc()
    p.inst(X(q2))

    sb = or_all(p, [q3, q2, q1])

    print p.out()
    print p.get_qubits()

    cxn = Connection()
    result = cxn.run_and_measure(p, [sb.index()])

    print result

def test_add_reversible_function(x):
    p = pq.Program()

    q1 = p.alloc()
    q2 = p.alloc()
    q3 = p.alloc()
    qubits = [q3, q2, q1]
    num_qubits = len(qubits)

    n = x
    for i in xrange(num_qubits):
        if n % 2 == 1:
            p.inst(X(qubits[num_qubits - 1 - i]))
        n /= 2

    mappings = [1, 0, 3, 2, 5, 7, 4, 6]
    f = lambda j: mappings[j]
    add_reversible_function(p, f, qubits)
    p.out()

    cxn = Connection()
    result = cxn.run_and_measure(p, [q.index() for q in qubits])

    y = reduce(lambda prev, next: (prev << 1) + next, result[0], 0)
    print "-"*20
    print "Test for x = ", x
    print "Expected: ", f(x)
    print "Got: ", y
    print "-"*20

    assert f(x) == y

def test_four_bit_add_reversible_function(x):
    p = pq.Program()

    q1 = p.alloc()
    q2 = p.alloc()
    q3 = p.alloc()
    q4 = p.alloc()
    qubits = [q4, q3, q2, q1]
    num_qubits = len(qubits)

    n = x
    for i in xrange(num_qubits):
        if n % 2 == 1:
            p.inst(X(qubits[num_qubits - 1 - i]))
        n /= 2

    mappings = [13, 9, 7, 6,
                1, 0, 12, 5,
                4, 8, 14, 3,
                2, 11, 10, 15]

    f = lambda j: mappings[j]
    add_reversible_function(p, f, qubits)
    p.out()

    cxn = Connection()
    result = cxn.run_and_measure(p, [q.index() for q in qubits])

    y = reduce(lambda prev, next: (prev << 1) + next, result[0], 0)
    print "-"*20
    print "Test for x = ", x
    print "Expected: ", f(x)
    print "Got: ", y
    print "-"*20

    assert f(x) == y


if __name__=='__main__':
    for x in range(2**4):
        test_four_bit_add_reversible_function(x)