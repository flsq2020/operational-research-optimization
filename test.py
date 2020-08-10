import numpy as np
from simplex_algorithm import SimplexAlgorithm


# object  min Z = -3 * x1 + x2 + x3
# s.t.
#       x1 - 2 * x2 +     x3 <= -11
#  -4 * x1 +     x2 + 2 * x3 >= 3
#   2 * x1 -              x3 = -1
#   x_i >= 0 (i = 1,2,3)

#
# A = np.array([[1, -2, 1], [-4, 1, 2], [2, 0, -1]])
# B = np.array([11, 3, -1])
# Z = np.array([-3, 1, 1])  # min
# restrict = ['<=', '>=', '=']


# A = np.array([[-1, 2], [0, 1]])
# B = np.array([2, 3])
# Z = np.array([1, 2]) # max
# restrict = ['<=', '<=']

A = np.array([[1, 2, 2, 1, 0], [3, 4, 1, 0, 1]])
B = np.array([8, 7])
Z = np.array([5, 2, 3, -1, 1])    # max
restrict = ['=', '=']

if __name__ == '__main__':
    optimal = SimplexAlgorithm(A, B, Z, 'max', restrict).run()
    print(optimal)
