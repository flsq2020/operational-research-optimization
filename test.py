import numpy as np
from simplex_algorithm import SimplexAlgorithm


# object  Z = -3 * x1 + x2 + x3
# s.t.
#       x1 - 2 * x2 +     x3 <= -11
#  -4 * x1 +     x2 + 2 * x3 >= 3
#   2 * x1 -              x3 = -1
#   x_i >= 0 (i = 1,2,3)


A = np.array([[1, -2, 1], [-4, 1, 2], [2, 0, -1]])
B = np.array([11, 3, -1])
Z = np.array([-3, 1, 1])
restrict = ['<=', '>=', '=']


if __name__ == '__main__':
    optimal = SimplexAlgorithm(A, B, Z, restrict).run()
    print(optimal)
