import numpy as np
import random
import copy
import time
from GA_comparison import GA
from matplotlib import pyplot as plt
#font=font_manager.FontProperties(fname="C:\Windows\Fonts\simhei.ttf",size=7.0)
def init():
    ammo = np.array([9, 9, 9])
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
    print(p_mtoa)
    return ammo, c, attacker, p_mtoa
def HM_init(ammo, attacker):
    indi = []
    #print(len(ammo), len(attacker))
    for i in range(len(ammo)):
        sum = 0
        for j in range(len(attacker[0])):
            x = random.randint(0, np.max(ammo[i]) - sum)
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
        value = -(p_yu + np.exp(-cost / 1000))
    else:
        value = -f
    return value

def sort_u(u, p_mtoa, ammo, c, len_u=5, p_yu=0.95):
    value = []
    sorted_u = []
    for i in u:
        value.append(indi_value(i, c, ammo, p_mtoa))
    for i in np.argsort(value):
        sorted_u.append(u[i])
    res = sorted_u[:len_u:]
    return res
def new_H(u, ammo, attacker,cont, iter, HMCR=0.95):
    temp = random.random()
    PAR_min, PAR_max = 0.1, 0.5
    PAR = PAR_max - (PAR_max - PAR_min) * cont / iter
    if temp < HMCR:
        new_indi = []
        #print(u[0])
        for i in range(len(u[0])):
            x = u[random.randint(0, len(u) - 1)][i]
            if random.random() < PAR:
                x = u[0][random.randint(0, len(u[0]) - 1)]
            new_indi.append(max(x, 0))
    else:
        new_indi = HM_init(ammo, attacker)
    return new_indi
def IGHS(ammo, c, attacker, p_mtoa, iteration = 500, p_yu = 0.95):
    time_start = time.time()

    f = []
    u_ori = []
    u_scale = 10
    for i in range(u_scale):
        indi = HM_init(ammo, attacker)
        f.append(eval(p_mtoa, indi, ammo))
        u_ori.append(indi)


    u = u_ori
    i = 0
    plot_x = []
    plot_y = []
    plot_z = []
    while(i < iteration):
        i += 1
        new_indi = new_H(u, ammo, attacker, i, iteration)
        u.append(new_indi)
        sorted_u = sort_u(u, p_mtoa, ammo, c, len_u=3)
        u = sorted_u
        good_indi = sorted_u[0]
        good_f = eval(p_mtoa, good_indi, ammo)
        good_cost = consumption(good_indi, c, ammo)
        plot_x.append(i)
        plot_y.append(good_f)
        plot_z.append(good_cost)
        #if i % 100 == 0:
            #print('iteration:', i+1, 'good_f:', good_f, 'good_cost', good_cost)
    time_end = time.time()
    time_consu = time_end - time_start
    plt.figure(1)
    plt.plot(plot_x, plot_y, color='blue', linestyle='-', label='本文方法')
    plt.xlabel('迭代次数')
    plt.ylabel('平均毁伤概率')
    plt.legend(loc='best')
    plt.figure(2)
    plt.plot(plot_x, plot_z, color='blue', linestyle='-', label='本文方法')
    #plt.axhline(y=3000, color='r', linestyle='-')
    plt.xlabel('迭代次数')
    plt.ylabel('成本消耗')
    plt.legend(loc='best')

    return good_indi, good_f, good_cost, time_consu





'''ammo, c, attacker, p_mtoa = init()
ammo = ammo.reshape(3, -1)
c = c.reshape(-1, 3)
attacker = attacker.reshape(-1, len(attacker))
p_yu = 0.95
HS_indi, HS_f, HS_cost, HS_time = GHS(ammo, c, attacker, p_mtoa, iteration=500)
HS_indi = np.array(HS_indi).reshape(len(ammo), -1)
print('和声搜索火力分配方案为:')
print(HS_indi)
if HS_f >= p_yu:
    print('和声搜索平均毁伤概率为：', str(HS_f)+str('>=')+str(p_yu))
else:
    print('和声搜索平均毁伤概率为：', HS_f)
print('和声搜索成本消耗为：', HS_cost)
print(str('和声搜索总耗时为：') + str(HS_time) + str('s'))


plt.show()'''