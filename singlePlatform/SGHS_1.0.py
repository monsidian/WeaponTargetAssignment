import numpy as np
import random
import copy
import time
import math
from GA_comparison import GA
from HS_mm_2 import HS
from PSO_mm import PSO
from IHS_mm import IHS
from GHS_mm import GHS
from IGHS_mm import IGHS
from CE_CAPSO_mm import CE_CAPSO
from matplotlib import pyplot as plt
#font=font_manager.FontProperties(fname="C:\Windows\Fonts\simhei.ttf",size=7.0)
def init():
    #return init_simple()
    #return init_complex()
    return init_most_complex()
def init_simple():
    ammo = np.array([10, 10, 10])
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
    ammo = np.array([3, 3, 3, 3])
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
    ammo = np.array([30, 30, 30, 30, 30])
    c = np.array([100, 120, 130, 125, 110])
    attacker = np.array(['A', 'B', 'C', 'B', 'A', 'C', 'A'])
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
def new_H(u, ammo, attacker,cont, iter, HMCR_sum, PAR_sum, HMCRm, PARm):
    temp = random.random()
    HMCR = np.random.normal(HMCRm, 0.01)
    PAR = np.random.normal(PARm, 0.05)
    HMCR_sum += HMCR
    PAR_sum += PAR

    band_min, band_max = 0.5, 1.5
    if cont >= iter / 2:
        band = band_min
    else:
        band = band_max - (band_max - band_min) * 2 * cont / iter
    #band = lambda cont, iter: band_min if cont >= iter / 2 else band_max - (band_max - band_min) * 2 * cont / iter
    if temp < HMCR:
        new_indi = []
        for i in range(len(u[0])):
            x = u[random.randint(0, len(u) - 1)][i] + 2 * band * random.randint(0, 1) - band
            if random.random() < PAR:
                x = u[0][i]
            new_indi.append(int(max(x, 0)))
    else:
        new_indi = HM_init(ammo, attacker)
    return new_indi, HMCR_sum, PAR_sum
def SGHS(ammo, c, attacker, p_mtoa, iteration = 1, p_yu = 0.95):
    time_start = time.time()
    LP = 100
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
        if not i % LP:
            PAR_sum = HMCR_sum = 0
            if not i:
                PARm, HMCRm = 0.9, 0.98
        if i % LP == LP - 1:
            HMCRm = HMCR_sum / LP
            PARm = PAR_sum / LP
        new_indi, HMCR_sum, PAR_sum = new_H(u, ammo, attacker, i, iteration, HMCR_sum, PAR_sum, HMCRm, PARm)
        u.append(new_indi)
        sorted_u = sort_u(u, p_mtoa, ammo, c, len_u=3)
        u = sorted_u
        good_indi = sorted_u[0]
        good_f = eval(p_mtoa, good_indi, ammo)
        good_cost = consumption(good_indi, c, ammo)
        plot_x.append(i)
        plot_y.append(good_f)
        plot_z.append(good_cost)
        i += 1
        #if i % 100 == 0:
            #print('iteration:', i+1, 'good_f:', good_f, 'good_cost', good_cost)
    time_end = time.time()
    time_consu = time_end - time_start
    plt.figure(1)
    plt.plot(plot_x, plot_y, color='green', label='基于SGHS的方法')
    plt.axhline(y=p_yu, color='black', label='概率阈值')
    plt.xlabel('迭代次数')
    plt.ylabel('平均毁伤概率')
    plt.legend(loc='best')
    plt.figure(2)
    plt.plot(plot_x, plot_z, color='green', label='基于SGHS的方法')
    #plt.axhline(y=3000, color='r', linestyle='-')
    plt.xlabel('迭代次数')
    plt.ylabel('成本消耗')
    plt.legend(loc='best')

    return good_indi, good_f, good_cost, time_consu





ammo, c, attacker, p_mtoa = init()
ammo = ammo.reshape(len(ammo), -1)
c = c.reshape(-1, len(ammo))
attacker = attacker.reshape(-1, len(attacker))
p_yu = 0.95
HS_time_sum = GA_time_sum = CE_CAPSO_time_sum = GHS_time_sum = IGHS_time_sum = IHS_time_sum = SGHS_time_sum = 0
HS_cost_sum = GA_cost_sum = CE_CAPSO_cost_sum = GHS_cost_sum = IGHS_cost_sum = IHS_cost_sum = SGHS_cost_sum = 0
HS_f_sum = GA_f_sum = CE_CAPSO_f_sum = GHS_f_sum = IGHS_f_sum = IHS_f_sum = SGHS_f_sum = 0

loop_cont = 1
for i in range(loop_cont):
    HS_indi, HS_f, HS_cost, HS_time = HS(ammo, c, attacker, p_mtoa, iteration=500)
    GA_indi, GA_f, GA_cost, GA_time =GA(ammo, c, attacker, p_mtoa, iteration=500)
    CE_CAPSO_indi, CE_CAPSO_f, CE_CAPSO_cost, CE_CAPSO_time = CE_CAPSO(ammo, c, attacker, p_mtoa, iteration=500)
    IHS_indi, IHS_f, IHS_cost, IHS_time = IHS(ammo, c, attacker, p_mtoa, iteration=500)
    GHS_indi, GHS_f, GHS_cost, GHS_time = GHS(ammo, c, attacker, p_mtoa, iteration=500)
    IGHS_indi, IGHS_f, IGHS_cost, IGHS_time = IGHS(ammo, c, attacker, p_mtoa, iteration=500)
    SGHS_indi, SGHS_f, SGHS_cost, SGHS_time = SGHS(ammo, c, attacker, p_mtoa, iteration=500)

    HS_indi = np.array(HS_indi).reshape(len(ammo), -1)
    GA_indi = np.array(GA_indi).reshape(len(ammo), -1)
    CE_CAPSO_indi = np.array(CE_CAPSO_indi).reshape(len(ammo), -1)
    IHS_indi = np.array(IHS_indi).reshape(len(ammo), -1)
    SGHS_indi = np.array(SGHS_indi).reshape(len(ammo), -1)
    GHS_indi = np.array(GHS_indi).reshape(len(ammo), -1)
    IGHS_indi = np.array(IGHS_indi).reshape(len(ammo), -1)



    HS_time_sum += HS_time
    GA_time_sum += GA_time
    CE_CAPSO_time_sum += CE_CAPSO_time
    IHS_time_sum += IHS_time
    GHS_time_sum += GHS_time
    IGHS_time_sum += IGHS_time
    SGHS_time_sum += SGHS_time

    HS_cost_sum += HS_cost
    GA_cost_sum += GA_cost
    CE_CAPSO_cost_sum += CE_CAPSO_cost
    IHS_cost_sum += IHS_cost
    GHS_cost_sum += GHS_cost
    IGHS_cost_sum += IGHS_cost
    SGHS_cost_sum += SGHS_cost

    HS_f_sum += HS_f
    GA_f_sum += GA_f
    CE_CAPSO_f_sum += CE_CAPSO_f
    IHS_f_sum += IHS_f
    GHS_f_sum += GHS_f
    IGHS_f_sum += IGHS_f
    SGHS_f_sum += SGHS_f

HS_time_ave = HS_time_sum / loop_cont
GA_time_ave = GA_time_sum / loop_cont
CE_CAPSO_time_ave = CE_CAPSO_time_sum / loop_cont
IHS_time_ave = IHS_time_sum / loop_cont
GHS_time_ave = GHS_time_sum / loop_cont
IGHS_time_ave = IGHS_time_sum / loop_cont
SGHS_time_ave = SGHS_time_sum / loop_cont

HS_cost_ave = HS_cost_sum / loop_cont
GA_cost_ave = GA_cost_sum / loop_cont
CE_CAPSO_cost_ave = CE_CAPSO_cost_sum / loop_cont
IHS_cost_ave = IHS_cost_sum / loop_cont
GHS_cost_ave = GHS_cost_sum / loop_cont
IGHS_cost_ave = IGHS_cost_sum / loop_cont
SGHS_cost_ave = SGHS_cost_sum / loop_cont

HS_f_ave = HS_f_sum / loop_cont
GA_f_ave = GA_f_sum / loop_cont
CE_CAPSO_f_ave = CE_CAPSO_f_sum / loop_cont
IHS_f_ave = IHS_f_sum / loop_cont
GHS_f_ave = GHS_f_sum / loop_cont
IGHS_f_ave = IGHS_f_sum / loop_cont
SGHS_f_ave = SGHS_f_sum / loop_cont

'''print(str('遗传算法平均耗时为：') + str(GA_time_ave) + str('s'))
print(str('改进鲇鱼效应——云自适应粒子群算法平均耗时为：') + str(CE_CAPSO_time_ave) + str('s'))
print(str('和声搜索算法平均耗时为：') + str(HS_time_ave) + str('s'))
print(str('改进和声搜索算法平均耗时为：') + str(IHS_time_ave) + str('s'))
print(str('全局最优和声搜索算法平均耗时为：') + str(GHS_time_ave) + str('s'))
print(str('改进全局最优和声搜索算法平均耗时为：') + str(IGHS_time_ave) + str('s'))
print(str('自适应全局最优和声搜索算法平均耗时为：') + str(SGHS_time_ave) + str('s'))
print('\n')
print('遗传算法平均毁伤概率为：', str(GA_f_ave))
print('改进鲇鱼效应——云自适应粒子群算法平均毁伤概率为：', str(CE_CAPSO_f_ave))
print('和声搜索算法平均毁伤概率为：', str(HS_f_ave))
print('改进和声搜索算法平均毁伤概率为：', str(IHS_f_ave))
print('全局最优和声搜索算法平均毁伤概率为：', str(GHS_f_ave))
print('改进全局最优和声搜索算法平均毁伤概率为：', str(IGHS_f_ave))
print('自适应全局最优和声搜索算法平均毁伤概率为：', str(SGHS_f_ave))
print('\n')
print('遗传算法平均成本消耗为：', str(GA_cost_ave))
print('改进鲇鱼效应——云自适应粒子群算法平均成本消耗为：', str(CE_CAPSO_cost_ave))
print('和声搜索算法平均成本消耗为：', str(HS_cost_ave))
print('改进和声搜索算法平均成本消耗为：', str(IHS_cost_ave))
print('全局最优和声搜索算法平均成本消耗为：', str(GHS_cost_ave))
print('改进全局最优和声搜索算法平均成本消耗为：', str(IGHS_cost_ave))
print('自适应全局最优和声搜索算法平均成本消耗为：', str(SGHS_cost_ave))
'''
print('遗传算法搜索火力分配方案为:')
print(GA_indi)
if GA_f >= p_yu:
    print('遗传算法平均毁伤概率为：', str(GA_f)+str('>=')+str(p_yu))
else:
    print('遗传算法平均毁伤概率为：', GA_f)
print('遗传算法成本消耗为：', GA_cost)
print(str('遗传算法总耗时为：') + str(GA_time) + str('s'))

print('改进鲇鱼效应——云自适应粒子群算法搜索火力分配方案为:')
print(CE_CAPSO_indi)
if CE_CAPSO_f >= p_yu:
    print('改进鲇鱼效应——云自适应粒子群算法平均毁伤概率为：', str(CE_CAPSO_f)+str('>=')+str(p_yu))
else:
    print('改进鲇鱼效应——云自适应粒子群算法平均毁伤概率为：', CE_CAPSO_f)
print('改进鲇鱼效应——云自适应粒子群算法成本消耗为：', CE_CAPSO_cost)
print(str('改进鲇鱼效应——云自适应粒子群算法总耗时为：') + str(CE_CAPSO_time) + str('s'))

print('和声搜索火力分配方案为:')
print(HS_indi)
if HS_f >= p_yu:
    print('和声搜索平均毁伤概率为：', str(HS_f)+str('>=')+str(p_yu))
else:
    print('和声搜索平均毁伤概率为：', HS_f)
print('和声搜索成本消耗为：', HS_cost)
print(str('和声搜索总耗时为：') + str(HS_time) + str('s'))



print('改进和声搜索火力分配方案为:')
print(IHS_indi)
if IHS_f >= p_yu:
    print('改进和声搜索平均毁伤概率为：', str(IHS_f)+str('>=')+str(p_yu))
else:
    print('改进和声搜索平均毁伤概率为：', IHS_f)
print('改进和声搜索成本消耗为：', IHS_cost)
print(str('改进和声搜索总耗时为：') + str(IHS_time) + str('s'))

print('全局和声搜索火力分配方案为:')
print(GHS_indi)
if GHS_f >= p_yu:
    print('全局和声搜索平均毁伤概率为：', str(GHS_f)+str('>=')+str(p_yu))
else:
    print('全局和声搜索平均毁伤概率为：', GHS_f)
print('全局和声搜索成本消耗为：', GHS_cost)
print(str('全局和声搜索总耗时为：') + str(GHS_time) + str('s'))

print('全局自适应最优和声搜索火力分配方案为:')
print(SGHS_indi)
if SGHS_f >= p_yu:
    print('全局自适应最优和声搜索平均毁伤概率为：', str(SGHS_f)+str('>=')+str(p_yu))
else:
    print('全局自适应最优和声搜索平均毁伤概率为：', SGHS_f)
print('全局自适应最优和声搜索成本消耗为：', SGHS_cost)
print(str('全局自适应最优和声搜索总耗时为：') + str(SGHS_time) + str('s'))

print('改进全局和声搜索火力分配方案为:')
print(IGHS_indi)
if IGHS_f >= p_yu:
    print('改进全局和声搜索平均毁伤概率为：', str(IGHS_f)+str('>=')+str(p_yu))
else:
    print('改进全局和声搜索平均毁伤概率为：', IGHS_f)
print('改进全局和声搜索成本消耗为：', IGHS_cost)
print(str('改进全局和声搜索总耗时为：') + str(IGHS_time) + str('s'))

plt.show()