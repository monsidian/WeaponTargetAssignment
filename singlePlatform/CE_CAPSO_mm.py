import numpy as np
import random
import copy
import time
from GA_comparison import GA
from matplotlib import pyplot as plt
from HS_mm_2 import HS
from matplotlib import pyplot as plt
def init():
    #return init_simple()
    #return init_complex()
    return init_most_complex()
def init_simple():
    ammo = np.array([7, 7, 7])
    c = np.array([100, 120, 130])
    attacker = np.array(['A', 'B', 'C'])
    p_ini = np.array([0.47, 0.44, 0.51, 0.46, 0.52, 0.54, 0.52, 0.53, 0.52])
    f_mtoa = np.empty(shape=(3, 0))
    p_ini = p_ini.reshape(3, 3)
    for i in attacker:
        if i == 'A':
            f_mtoa = np.column_stack((f_mtoa, [1, 0, 0]))
        if i == 'B':
            f_mtoa = np.column_stack((f_mtoa, [0, 1, 0]))
        if i == 'C':
            f_mtoa = np.column_stack((f_mtoa, [0, 0, 1]))
    p_mtoa = np.dot(p_ini, f_mtoa)
    print('毁伤概率表为：')
    print(p_mtoa)
    return ammo, c, attacker, p_mtoa
def init_complex():
    ammo = np.array([9, 9, 9, 9])
    c = np.array([100, 120, 130, 125])
    attacker = np.array(['A', 'B', 'C', 'B'])
    p_ini = np.array([0.47, 0.44, 0.51, 0.46, 0.52, 0.54, 0.52, 0.53, 0.52, 0.48, 0.51, 0.54])
    f_mtoa = np.empty(shape=(3, 0))
    p_ini = p_ini.reshape(len(c), 3)
    #print(p_ini)
    for i in attacker:
        if i == 'A':
            f_mtoa = np.column_stack((f_mtoa, [1, 0, 0]))
        if i == 'B':
            f_mtoa = np.column_stack((f_mtoa, [0, 1, 0]))
        if i == 'C':
            f_mtoa = np.column_stack((f_mtoa, [0, 0, 1]))
    p_mtoa = np.dot(p_ini, f_mtoa)
    print('毁伤概率表为：')
    print(p_mtoa)
    return ammo, c, attacker, p_mtoa
def init_most_complex():
    ammo = np.array([15, 15, 15, 15, 15])
    c = np.array([100, 120, 130, 125, 110])
    attacker = np.array(['A', 'B', 'C', 'B', 'A', 'C'])
    p_ini = np.array([0.47, 0.44, 0.51, 0.46, 0.52, 0.54, 0.52, 0.53, 0.52, 0.48, 0.51, 0.54, 0.49, 0.51, 0.53]) #弹药种类量 * 来袭目标的种类量
    f_mtoa = np.empty(shape=(3, 0))
    p_ini = p_ini.reshape(len(c), 3)
    #print(p_ini)
    for i in attacker:
        if i == 'A':
            f_mtoa = np.column_stack((f_mtoa, [1, 0, 0]))
        if i == 'B':
            f_mtoa = np.column_stack((f_mtoa, [0, 1, 0]))
        if i == 'C':
            f_mtoa = np.column_stack((f_mtoa, [0, 0, 1]))
    p_mtoa = np.dot(p_ini, f_mtoa)
    print('毁伤概率表为：')
    print(p_mtoa)
    return ammo, c, attacker, p_mtoa

def group_init(ammo, attacker):
    indi = []
    #print(len(ammo), len(attacker))
    for i in range(len(ammo)):
        sum = 0
        for j in range(len(attacker[0])):
            x = random.randint(0, np.max(ammo[i]) - sum)
            #x = random.randint(0, 1)
            #x = random.randint(0,1)
            sum += x
            indi.append(x)
    '''for i in range(len(ammo) * len(attacker[0])):
        x = random.randint(0, np.max(ammo))
        indi.append(x)'''
    #print(len(attacker[0]))
    #print(max(ammo[0]))
    return indi

def eval(p_mtoa, indi, ammo, p_yu=0.95):
    indi = np.array(indi).reshape(len(ammo), -1)
    for i in range(len(ammo)):
        if np.sum(indi, axis=1)[i] > ammo[i]:
            #print('res 0', aij)
            return 0
    p = 1 - np.power(1 - p_mtoa, indi)
    q = 1 - np.prod(1 - p, axis=0)
    res = np.sum(q) / len(indi[0])
    #res = max(p_yu, res)
    #return min(res, p_yu)
    return res
def consumption(indi, c, ammo):
    indi_c = np.array(copy.deepcopy(indi)).reshape(len(ammo), -1)
    cost = np.sum(np.sum(indi_c, axis=1) * c)
    return cost
def indi_value(indi, c, ammo, p_mtoa, p_yu=0.95):
    f = eval(p_mtoa, indi, ammo)
    if(f >= p_yu):
        cost = consumption(indi, c, ammo)
        value = (p_yu + np.exp(-cost / 1000))
    else:
        value = f
    return value

def PSO_updata(u, v, f, u_scale, indi_scale, u_indi_max, best_indi, c, ammo, p_mtoa, f_ave, f_ave_nic, f_ave_bad):
    alpha, beta = 1, 0.12
    c1, c2 = 2, 2
    n = 2
    ce_v = 0.5 * pow(n, 0.5)
    u_temp = []
    u_indi_max_up = copy.deepcopy(u_indi_max)
    for i in range(u_scale):
        indi_temp = []
        for j in range(indi_scale):
            #print(u[i])
            r1 = random.random()
            r2 = random.random()
            if u[i][j] == best_indi[j]:
                if f[i] > f_ave_nic:
                    w = 0.4
                    v[i][j] = w * v[i][j] + c2 * r2 * (best_indi[j] - u[i][j])
                elif f[i] > f_ave:
                    ex = max(f)
                    en = (f_ave_nic - ex) / alpha
                    he = en / beta
                    en1 = np.random.normal(en, he * he)
                    w = 0.9 - 0.5 * np.exp(-0.5 * pow(f[i] - ex, 2) / pow(en1, 2))
                    v[i][j] = w * v[i][j] + c1 * r1 * (u_indi_max[i][j] - u[i][j]) + c2 * r2 * (best_indi[j] - u[i][j])
                else:
                    w = 0.9
                    v[i][j] = w * v[i][j] + c1 * r1 * (u_indi_max[i][j] - u[i][j])
            else:
                if f[i] > f_ave_nic:
                    w = 0.4
                    v[i][j] = w * v[i][j] + c2 * r2 * (best_indi[j] - u[i][j]) + ce_v
                elif f[i] > f_ave:
                    ex = max(f)
                    en = (f_ave_nic - ex) / alpha
                    he = en / beta
                    #print(en, he * he)
                    en1 = np.random.normal(en, he * he)
                    w = 0.9 - 0.5 * np.exp(-0.5 * pow(f[i] - ex, 2) / pow(en1, 2))
                    v[i][j] = w * v[i][j] + c1 * r1 * (u_indi_max[i][j] - u[i][j]) + c2 * r2 * (best_indi[j] - u[i][j]) + ce_v
                else:
                    w = 0.9
                    v[i][j] = w * v[i][j] + c1 * r1 * (u_indi_max[i][j] - u[i][j]) + ce_v
            x = u[i][j] + int(v[i][j])
            indi_temp.append(max(0, x))
        #print(indi_value(indi_temp, c, ammo, p_mtoa), f[i])
        if indi_value(indi_temp, c, ammo, p_mtoa) > f[i]:
            u_indi_max_up[i] = indi_temp
        u_temp.append(indi_temp)
    #print(u_temp)
    return u_temp, v, u_indi_max_up






def CE_CAPSO(ammo, c, attacker, p_mtoa, iteration = 1, p_yu = 0.95):
    time_start = time.time()

    indi_scale = len(ammo) * len(attacker[0])

    f = []
    value = []
    u_ori = []
    u_scale = 10
    for i in range(u_scale):
        indi = group_init(ammo, attacker)

        f.append(indi_value(indi, c, ammo, p_mtoa))
        u_ori.append(indi)

    u = u_ori
    ind_max = np.argmax(f)
    best_indi = u[ind_max]


    u_indi_max = copy.deepcopy(u)
    i = 0
    v = [[0] * indi_scale for _ in range(u_scale)]
    plot_x = []
    plot_y = []
    plot_z = []
    while (i < iteration):

        f_ave = np.average(f)
        con_bad, con_nic = 0, 0
        f_bad, f_nic = 0, 0
        for k in range(len(f)):
            if f[k] > f_ave:
                con_nic += 1
                f_nic += 1
            else:
                con_bad += 1
                f_bad += 1
        f_ave_bad = f_bad / con_bad
        f_ave_nic = f_nic / con_nic

        u_up, v, u_indi_max_up = PSO_updata(u, v, f, u_scale, indi_scale, u_indi_max, best_indi, c, ammo, p_mtoa, f_ave, f_ave_nic, f_ave_bad)
        for j in range(u_scale):
            if u_indi_max[j] != u_indi_max_up[j]:
                #print(u_indi_max_up[j])
                u_indi_max[j] = u_indi_max_up[j]
                u[j] = u_up[j]
                f[j] = indi_value(u_up[j], c, ammo, p_mtoa)


        ind_max = np.argmax(f)
        best_indi = u[ind_max]
        p_hui = eval(p_mtoa, best_indi, ammo)
        c_min = consumption(best_indi, c, ammo)
        plot_x.append(i)
        plot_y.append(p_hui)
        plot_z.append(c_min)

        i += 1

    plt.figure(1)
    plt.plot(plot_x, plot_y, color='red', label='基于改进CE-CAPSO的方法', linestyle='--')
    plt.xlabel('迭代次数')
    plt.ylabel('平均毁伤概率')
    plt.legend(loc='best')
    plt.figure(2)
    plt.plot(plot_x, plot_z, color='red', label='基于改进CE-CAPSO的方法', linestyle='--')
    plt.xlabel('迭代次数')
    plt.ylabel('成本消耗')
    plt.legend(loc='best')

    time_end = time.time()
    time_consu = time_end - time_start
    return best_indi, p_hui, c_min, time_consu


'''ammo, c, attacker, p_mtoa = init()
attacker = attacker.reshape(-1, len(attacker))
p_yu = 0.95
CE_CAPSO_indi, CE_CAPSO_f, CE_CAPSO_cost, CE_CAPSO_time = CE_CAPSO(ammo, c, attacker, p_mtoa, iteration=500)

CE_CAPSO_indi = np.array(CE_CAPSO_indi).reshape(len(ammo), -1)

print('粒子群算法搜索火力分配方案为:')
print(CE_CAPSO_indi)
if CE_CAPSO_f >= p_yu:
    print('粒子群算法平均毁伤概率为：', str(CE_CAPSO_f)+str('>=')+str(p_yu))
else:
    print('粒子群算法平均毁伤概率为：', CE_CAPSO_f)
print('粒子群算法成本消耗为：', CE_CAPSO_cost)
print(str('粒子群算法总耗时为：') + str(CE_CAPSO_time) + str('s'))
plt.show()'''