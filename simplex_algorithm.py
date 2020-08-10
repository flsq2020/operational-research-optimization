import numpy as np
import warnings
warnings.filterwarnings("ignore")


class SimplexAlgorithm:
    def __init__(self, A, B, Z, mode, restrict, x_restrict=None):
        self.A = A
        self.B = B
        self.Z = Z
        self.mode = mode
        self.restrict = restrict
        self.x_restrict = x_restrict
        self.param = []
        self.optimal = 0

    def __str__(self):
        return 'SimplexAlgorithm'

    def run(self):
        if check_standard_linear_programming(self.B, self.restrict):
            self.optimal = simplex_algorithm(self.A, self.B, self.Z, self.mode)
        else:
            A, B, Z = transform_standard_linear_programming(self.A, self.B, self.Z, self.restrict)
            self.optimal = simplex_algorithm(A, B, Z, self.mode)
        return self.optimal


def check_standard_linear_programming(b, restrict):
    check_b = all(map(lambda x: True if x >= 0 else False, b))
    if restrict:
        check_restrict = all(map(lambda x: True if x == '=' else False, restrict))
        return check_restrict and check_b
    else:
        return check_b


def transform_standard_linear_programming(a, b, z, restrict):
    slack_coefficient = lambda x: {'>=': -1, '<=': 1, '=': 0}.get(x)
    is_slack = lambda x: x[0] if x[1] == 0 else -1
    equations = len(a)
    add_slack = np.zeros((equations, equations))
    slack_coeff = list(map(slack_coefficient, restrict))
    mask = [True] * equations
    add_slack[mask, mask] = slack_coeff
    slack = list(map(is_slack, enumerate(slack_coeff)))
    add_slack = np.delete(add_slack, slack, axis=1)
    a_standard = np.hstack([a, add_slack])
    b_negative = list(filter(lambda x: True if x[1] < 0 else False, enumerate(b)))
    if len(b_negative) > 0:
        b_negative = [b[0] for b in b_negative]
        a_standard[b_negative,:] = -1 * a_standard[b_negative,:]
    if isinstance(z, np.ndarray):
        z = z.tolist()
    return a_standard, np.abs(b), z + [0] * slack.count(-1)


def search_basic_variable(a):
    func = lambda x: True if len(x[1][0]) == 0 and x[1][1] == 1 else False
    # 找到所有基变量所在列
    basic_col_index = list(filter(func, enumerate([(set(row) - {0, 1}, sum(row))for row in a.T])))
    basic_col_index = [col_index[0] for col_index in basic_col_index]
    # 找到所有基变量所在行
    basic_row_index = np.array([np.argwhere(a == 1).flatten() for a in a.T])[basic_col_index]
    basic_row_index = [row_index[0] for row_index in basic_row_index]
    basic = list(zip(basic_row_index, basic_col_index))
    return sorted(basic, key=lambda x: x[0])


def basic_new_old_by_check_number(a, b, check, basic_table, mode):
    flag = True
    func = {'min': min, 'max': max}
    m_value = 1e6
    new = check.index(func[mode](check))
    if b.ndim == 2:
        b = np.squeeze(b)
    b_div_a = b / a[:, new]
    if all(a[:, new] <= 0):
        flag = False
    b_div_a[np.isinf(b_div_a)] = m_value
    b_div_a[b_div_a <= 0] = m_value
    out = b_div_a.tolist().index(min(b_div_a))
    print(dict(basic_table))
    old = dict(basic_table)[out]
    return new, old, out, flag


def rotation_transform(rotation, base_equ, basic_v):
    if rotation.dtype == np.int:
        rotation = rotation.astype('float')
    rotation[base_equ] = rotation[base_equ] / rotation[base_equ][basic_v]
    for i in range(len(rotation)):
        if i == base_equ:
            continue
        rotation[i] = rotation[i] - rotation[base_equ] * rotation[i][basic_v]
    return rotation


def update_basic(basic, out, in_b):
    equation = list(filter(lambda x: True if x[-1] == out else False, basic))[0][0]
    basic = list(filter(lambda x: False if x[-1] == out else True, basic))
    basic.append((equation, in_b))
    return sorted(basic, key=lambda x: x[0])


def simplex_algorithm(a, b, z, mode='min'):
    equations, variables = a.shape  # 方程数, 变量数
    # 判断是否每个方程都有基变量
    if len(search_basic_variable(a)) < equations:
        # 大M单纯形
        a, b, z, is_solution = bigMsimplex(a, b, z, mode)
        if not is_solution:
            return None
    if not isinstance(z, np.ndarray):
        z = np.array(z)
    a, b, z, object_v, _ = simplex(a, b, z, mode, '')
    return object_v


def bigMsimplex(a, b, z, mode):
    # 基变量索引
    basic_v_list = search_basic_variable(a)
    basic_v_row = [row[0] for row in basic_v_list]
    equations, variables = a.shape
    # 添加人工变量
    artificial = np.delete(np.eye(equations), basic_v_row, axis=1)
    a = np.hstack([a, artificial])
    M = 1
    w = np.hstack([np.zeros_like(z), np.array([M] * (equations - len(basic_v_list)))])
    # 单纯形
    a, b, w, _, is_solution = simplex(a, b, w, mode)
    # 删除人工变量
    a = a[:, :variables]
    return a, b, z, is_solution


def simplex(a, b, z, mode, stop_condition='M'):
    # 基变量索引
    basic_v_list = search_basic_variable(a)
    while True:
        is_solution = True
        print("===============")
        c_b = [v[-1] for v in basic_v_list]
        print('系数矩阵')
        print(a)
        print('基本变量', c_b)
        print('基本变量系数', z[c_b])
        # 目标值
        if isinstance(b, list):
            b = np.array(b)
        object_v = np.dot(z[c_b], b.reshape(-1, 1))
        print('目标函数值', object_v)
        # 计算所有检验数
        c = [(z[i] - np.dot(z[c_b], a[:, i:i+1]))[0] for i in range(a.shape[-1])]
        if stop_condition == 'M':
            if object_v == 0:
                print("人工变量已经全部成为非基本变量.")
                break
        else:
            if {'min': min(c) >= 0, 'max': max(c) <= 0}.get(mode):
                print("已经取得最优值.")
                break

        print('所有检验数', c)
        print('常数列', b)
        # 选最小检验数变量, 新变量入基，旧基变量出基
        new, old, out, flag = basic_new_old_by_check_number(a, b, c, basic_v_list, mode)
        if not flag:
            print('线性规划问题无解')
            is_solution = False
            object_v = None
            break
        print('新基本变量', new)
        print('被替换的基本变量', old)
        # 旋转变换
        rotation = np.hstack([a, b.reshape(-1, 1)])
        rotation = rotation_transform(rotation, out, new)
        print('rotation transform')
        print(rotation)
        a, b = rotation[:, :-1], rotation[:, -1:]
        # 更新basic_v_list
        basic_v_list = update_basic(basic_v_list, old, new)
        print('更新基变量坐标')
        print(basic_v_list)
    return a, b, z, object_v, is_solution