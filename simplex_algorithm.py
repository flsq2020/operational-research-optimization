import numpy as np
import warnings
warnings.filterwarnings("ignore")


class SimplexAlgorithm:
    def __init__(self, A, B, Z, restrict, x_restrict=None):
        self.A = A
        self.B = B
        self.Z = Z
        self.restrict = restrict
        self.x_restrict = x_restrict
        self.param = []
        self.optimal = 0

    def __str__(self):
        return 'SimplexAlgorithm'

    def run(self):
        if check_standard_linear_programming(self.B, self.restrict):
            self.optimal = simplex_algorithm(self.A, self.B, self.Z)
        else:
            A, B, Z = transform_standard_linear_programming(self.A, self.B, self.Z, self.restrict)
            self.optimal = simplex_algorithm(A, B, Z)
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
    is_slack = lambda x: x[0] if x[1]==0 else -1
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
    func = lambda x: True if len(x[1]) == 0 else False
    # 找到所有基变量所在列
    basic_col_index = list(filter(func, enumerate([set(row) - {0, 1} for row in a.T])))
    basic_col_index = [col_index[0] for col_index in basic_col_index]
    # 找到所有基变量所在行
    basic_row_index = np.array([np.argwhere(a == 1).flatten() for a in a.T])[basic_col_index]
    basic_row_index = [row_index[0] for row_index in basic_row_index]
    basic = list(zip(basic_row_index, basic_col_index))
    return sorted(basic, key=lambda x: x[0])


def basic_new_old_by_check_number(a, b, check, basic_table, mode='min'):
    func = {'min': min, 'max': max}
    m_value = {'min': 1e5, 'max': -1e5}

    new = check.index(func[mode](check))
    if b.ndim == 2:
        b = np.squeeze(b)
    b_div_a = b / a[:, new]
    b_div_a[np.isinf(b_div_a)] = m_value[mode]
    b_div_a[b_div_a <= 0] = m_value[mode]
    out = b_div_a.tolist().index(func[mode](b_div_a))
    old = dict(basic_table)[out]
    return new, old, out


def rotation_transform(rotation, base_equ, basic_v):
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


def simplex_algorithm(a, b, z):
    equations, variables = a.shape  # 方程数, 变量数
    # 判断是否每个方程都有基变量
    if len(search_basic_variable(a)) < equations:
        # 大M单纯形
        a, b, z = bigMsimplex(a, b, z)
        if not isinstance(z, np.ndarray):
            z = np.array(z)
        basic_v_list = search_basic_variable(a)
        # 单纯形
        a, b, z, object_v = simplex(a, b, z, basic_v_list, stop_condition=2)
    else:
        pass
    return object_v


def bigMsimplex(a, b, z):
    basic_v_list = search_basic_variable(a)
    basic_v_row = [row[0] for row in basic_v_list]
    equations, variables = a.shape
    # 添加人工变量
    artificial = np.delete(np.eye(equations), basic_v_row, axis=1)
    artificial_row = list({i for i in range(equations)} - set(basic_v_row))
    artificial_col = [i + variables for i in range(artificial.shape[-1])]
    a = np.hstack([a, artificial])
    M = 1
    w = np.hstack([np.zeros_like(z), np.array([M] * (equations - len(basic_v_list)))])
    # 基变量索引
    basic_v_list = basic_v_list + list(zip(artificial_row, artificial_col))
    # 单纯形
    a, b, w, _ = simplex(a, b, w, basic_v_list, 1)
    # 删除人工变量
    a = a[:, :variables]
    return a, b, z


def simplex(a, b, w, basic_v_list,stop_condition=1):
    while True:
        print("===============")
        c_b = [v[-1] for v in basic_v_list]
        print('系数矩阵')
        print(a)
        print('基本变量',c_b)
        print('基本变量系数',w[c_b])
        # 目标值
        object_v = np.dot(w[c_b], b.reshape(-1,1))
        print('目标函数值',object_v)
        c = [(w[i] - np.dot(w[c_b], a[:,i:i+1]))[0] for i in range(a.shape[-1])]
        if stop_condition == 1:
            if object_v == 0:
                print("人工变量已经全部成为非基本变量.")
                break
        else:
            if min(c) >= 0:
                print("已经取得最优值.")
                break
        # 计算所有检验数
        c = [(w[i] - np.dot(w[c_b], a[:,i:i+1]))[0] for i in range(a.shape[-1])]
        print('所有检验数',c)
        print('常数列',b)
        # 选最小检验数变量, 新变量入基，旧基变量出基
        new, old, out = basic_new_old_by_check_number(a, b, c, basic_v_list)
        print('新基本变量',new)
        print('被替换的基本变量',old)
        # 旋转变换
        rotation = np.hstack([a,b.reshape(-1,1)])
        rotation = rotation_transform(rotation, out, new)
        print('rotation transform')
        print(rotation)
        a, b = rotation[:,:-1], rotation[:,-1:]
        print('更新基变量坐标')
        print(update_basic(basic_v_list, old, new))
        # 更新basic_v_list
        basic_v_list = update_basic(basic_v_list, old, new)
    return a, b, z, object_v