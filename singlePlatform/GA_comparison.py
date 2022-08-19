import numpy as np
import random
import copy
import time
from matplotlib import pyplot as plt
#font=font_manager.FontProperties(fname="C:\Windows\Fonts\simhei.ttf",size=7.0)
def init():
    ammo = np.array([10, 10, 10])
    c = np.array([100, 120, 130])
    attacker = np.array(['A', 'B', 'C', 'A'])
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
def consumption(indi, c, ammo):
    indi_c = np.array(copy.deepcopy(indi)).reshape(len(ammo), -1)
    cost = np.sum(np.sum(indi_c, axis=1) * c)
    return cost
def eval(p_mtoa, indi, ammo, c,p_yu=0.95):
    #print(a)
    #print(b)
    #print(len(indi))
    indi = np.array(indi).reshape(len(ammo), -1)
    #print('ammo:', len(ammo))
    #print('indi', len(indi[0]))
    for i in range(len(ammo)):
        if np.sum(indi, axis=1)[i] > ammo[i]:
            #print('res 0', aij)
            return 0.0001 * random.random()
    p = 1 - np.power(1 - p_mtoa, indi)
    q = 1 - np.prod(1 - p, axis=0)
    res = np.sum(q) / len(indi[0])
    if(res >= p_yu):
        cost = consumption(indi, c, ammo)
        value = p_yu + np.exp(-cost / 1000)
    else:
        value = res
    return value
def consumption(indi, c, ammo):
    indi_c = np.array(copy.deepcopy(indi)).reshape(len(ammo), -1)
    cost = np.sum(np.sum(indi_c, axis=1) * c)
    return cost
def p_eval(p_mtoa, indi, ammo, c,p_yu=0.95):
    #print(a)
    #print(b)
    #print(len(indi))
    indi = np.array(indi).reshape(len(ammo), -1)
    #print('ammo:', len(ammo))
    #print('indi', len(indi[0]))
    for i in range(len(ammo)):
        if np.sum(indi, axis=1)[i] > ammo[i]:
            #print('res 0', aij)
            return 0.0001 * random.random()
    p = 1 - np.power(1 - p_mtoa, indi)
    q = 1 - np.prod(1 - p, axis=0)
    res = np.sum(q) / len(indi[0])
    return res
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
def group_p(f, u):
    F = sum(f)
    if F == 0:
        print('erro:', u)
        print('erro:', f)
    p = []
    q = []
    for i in range(len(f)):
        p.append(f[i] / F)
        q.append(sum(p))
    return p, q
def group_copy(u, q_acc):
    u_copy = []
    for i in range(len(u)):
        temp = random.random()
        for j in range(len(q_acc)):
            if temp < q_acc[j]:
                u_copy.append(u[j])
                #print(temp, j)
                break
    return u_copy
'''def strtobi(u):
    bi = [[] for j in range(len(u))]
    for i in range(len(u)):
        for j in range(len(u[i])):'''

def mating(a, b, sca = 3):
    par_mate = random.randint(1, len(a))
    #print('before mating:', a, b)
    #print('par_mate is',par_mate)
    for i in range(int(par_mate)):
        temp = a[i]
        a[i] = b[i]
        b[i] = temp
    '''if par_mate % sca == 1:
        if a[j] < pow(2, sca - 1) <= b[j]:
            a[j] += pow(2, sca - 1)
            b[j] -= pow(2, sca - 1)
        elif a[j] >= pow(2, sca - 1) > b[j]:
            a[j] -= pow(2, sca - 1)
            b[j] += pow(2, sca - 1)'''
    #print('after mating:', a, b)
    return a, b


def group_mate(u_copy):
    u_mate = copy.deepcopy(u_copy)
    temp = []
    for i in range(len(u_copy)):
        temp.append(random.random())
    cou = np.argsort(temp)
    u_mate[cou[0]], u_mate[cou[1]] = mating(u_copy[cou[0]], u_copy[cou[1]])
    return u_mate
def mutation(u_mate, max_ammo, p_mutation = 0.01):
    u_muta = copy.deepcopy(u_mate)
    temp = []
    chromosome = []
    gene = []
    for i in range(len(u_muta) * len(u_muta[0])):
        if random.random() < p_mutation:
            #print('haha')
            temp.append(i)
            chromosome.append(divmod(i, len(u_muta[0]))[0])
            gene.append(divmod(i, len(u_muta[0]))[1])
    #print(chromosome)
    #print(gene)
    for k in range(len(chromosome)):
        i = chromosome[k]
        j = gene[k]
        #u_muta[i][int(j)] = max_ammo - u_muta[i][int(j)]
        u_muta[i][int(j)] = random.randint(0, max_ammo)
    return u_muta
def GA(ammo, c, attacker, p_mtoa, iteration = 1, p_yu = 0.95):
    time_start = time.time()
    max_ammo = np.max(ammo)
    flag = 0
    while(flag == 0):
        #print('wawawawa')
        f = []
        u_ori = []
        for i in range(20):
            indi = group_init(ammo, attacker)
            f.append(eval(p_mtoa, indi, ammo, c))
            u_ori.append(indi)
            b = copy.deepcopy(indi)
            b = np.array(b).reshape(len(ammo), -1)
            for j in range(len(ammo)):
                if np.sum(b, axis=1)[j] > ammo[j]:
                    flag = 0
                    break
                else:
                    flag = 1
        #print('init_here', u_ori)
    #print('init_u:', u_ori)
    #print('init_f:', max(f))
    u = u_ori
    f_max = max(f)
    ind_max = np.argmax(f)
    best_u = u[ind_max]
    p_hui = p_eval(p_mtoa, best_u, ammo, c, p_yu)
    c_min = consumption(best_u, c, ammo)
    '''u_c = np.array(copy.deepcopy(u[ind_max])).reshape(len(ammo), -1)
    c_min = np.sum(np.sum(u_c, axis=1) * c)'''
    i = 0
    #print('first', best_u)
    plot_x = []
    plot_y = []
    plot_z = []
    while(i < iteration):
        p_copy, q_acc = group_p(f, u)
        u_copy = group_copy(u, q_acc)
        u_mate = group_mate(u_copy)
        u_muta = mutation(u_mate, max_ammo)
        u = u_muta
        f = []
        for j in range(20):
            f.append(eval(p_mtoa, u[j], ammo, c))
        if max(f) > f_max:
            f_max = max(f)
            ind_max = np.argmax(f)
            best_u = u[ind_max]
            p_hui = p_eval(p_mtoa, best_u, ammo, c, p_yu)
            c_min = consumption(best_u, c, ammo)
        '''if max(f) >= p_yu:
            f_max = max(f)
            ind_max = np.argmax(f)
            if cost < c_min:
                best_u = u[ind_max]
                c_min = cost
        elif max(f) >= f_max:
            f_max = max(f)
            ind_max = np.argmax(f)
            best_u = u[ind_max]
            c_min = cost'''
        plot_x.append(i)
        plot_y.append(p_hui)
        plot_z.append(c_min)
        '''plt.figure(1)
        plt.scatter(i, f_max)
        plt.figure(2)
        plt.scatter(i, c_min)'''
        '''if i % 100 == 0:
            bar_wid = 0.3
            plt.bar(i / 100, f_max, bar_wid, align='center', color = 'c', alpha=0.5)
            plt.bar((i / 100) + bar_wid, c_min / 10000, bar_wid, color='b', align='center', alpha=0.5)
            plt.xlabel('iteration/100')'''
        '''if i % 100 == 0:
            print('iteration:', i+1, 'f_max:', f_max)'''
        i += 1
    plt.figure(1)
    plt.plot(plot_x, plot_y, color='red', label='基于GA的方法', linestyle=':')
    plt.xlabel('迭代次数')
    plt.ylabel('平均毁伤概率')
    plt.legend(loc='best')
    plt.figure(2)
    plt.plot(plot_x, plot_z, color='red', label='基于GA的方法', linestyle=':')
    plt.xlabel('迭代次数')
    plt.ylabel('成本消耗')
    plt.legend(loc='best')
    #print(best_u, f_max)
    #return best_u, f_max
    time_end = time.time()
    time_consu = time_end - time_start
    return best_u, p_hui, c_min, time_consu








'''a = np.array([0.1, 0.2, 0.3, 0.4]).reshape(2, 2)
b = np.array([1, 2, 3, 4]).reshape(2, 2)
n = 1- b * np.log(1 - a)
print(n)'''
#print(sum(np.prod(1-b, axis=0)))
#w = np.array(v).reshape(-1, 1)
#print(w)
#print(type(u[0]))