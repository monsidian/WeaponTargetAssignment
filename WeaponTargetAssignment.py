import numpy as np
import copy


class WTA:
    def __init__(self, platForm, missile, attacker, pThreshold):  # 平台 火力单元 拦截目标 概率阈值
        self.platForm = platForm
        self.missile = missile
        self.attacker = attacker
        self.pThreshold = pThreshold
        self.dim = (platForm, missile, attacker)
        self.ammo = np.random.randint(0, 10, (self.platForm, self.missile))  # 平台火力单元库存
        self.cost = np.array([100 + 10 * i for i in range(self.missile)])  # 每种类型的火力单元的成本
        self.p_mtoa = np.random.normal(0.6, 0.1, (self.missile, self.attacker))  # 单枚火力单元对应拦截目标的毁伤概率
        self.constraint_ptoa = np.zeros((self.platForm, self.attacker)).astype('int')  # 平台对拦截目标约束，1表示不能攻击该目标
        for i in range(self.platForm):
            # self.constraint_ptoa[i, np.random.randint(0, self.missile)] = 1
            self.constraint_ptoa[i, i] = 1
        # print(self.constraint_ptoa)
        print(self.p_mtoa)
        # print(self.ammo)
        # print(self.cost)


class Individual:
    def __init__(self, chromosome, fitness, pDamage, consumption):
        self.chromosome = chromosome  # 染色体
        self.fitness = fitness  # 适应度
        self.pDamage = pDamage  # 毁伤概率
        self.consumption = consumption  # 成本消耗
        self.F = 0
        self.CR = 0

    def setCR(self, platForm, missile):
        self.CR = np.zeros((platForm, missile))


class Ele:
    def __init__(self, WTAInstance):
        self.WTAInstance = WTAInstance

    def IndGenerate(self):  # 初始化一个个体
        ammo = copy.deepcopy(self.WTAInstance.ammo)
        chro = np.zeros(self.WTAInstance.dim).astype('int')
        for k in range(self.WTAInstance.platForm):
            genPlat = chro[k]
            for i in range(self.WTAInstance.missile):
                for j in range(self.WTAInstance.attacker):
                    if self.WTAInstance.constraint_ptoa[k, j]:
                        genPlat[i, j] = 0
                        continue
                    genPlat[i, j] = np.random.randint(0, ammo[k, i] + 1)
                    ammo[k, i] -= genPlat[i, j]
        dic = self.GetFitness(chro)
        ind = Individual(chro, dic['fitness'], dic['pDamage'], dic['consumption'])
        return ind

    def GetFitness(self, ind):  # 求个体适应度
        # print(self.ammo)
        # print(indSum)
        for k in range(self.WTAInstance.platForm):
            for i in range(self.WTAInstance.missile):
                if np.sum(ind[k], axis=1)[i] > self.WTAInstance.ammo[k, i]:
                    return {'fitness': 0, 'pDamage': 0, 'consumption': float('inf')}
        pDamage = self.GetpDamage(ind)
        consumption = self.GetConsumption(ind)
        if pDamage < self.WTAInstance.pThreshold:
            fitness = pDamage
        else:
            fitness = self.WTAInstance.pThreshold + np.exp(-consumption / 10000)
        dic = {'fitness': fitness, 'pDamage': pDamage, 'consumption': consumption}
        return dic

    def GetpDamage(self, ind):  # 求毁伤概率
        indSum = np.sum(ind, axis=0)
        p = 1 - np.power(1 - self.WTAInstance.p_mtoa, indSum)
        q = 1 - np.prod(1 - p, axis=0)
        # print(q)
        pDamage = np.power(np.cumprod(q, axis=0)[-1], 1 / self.WTAInstance.attacker)  # 几何平均
        # if not pDamage:
        #     print(p)
        #     print(q)
        return pDamage

    def GetConsumption(self, ind):  # 求成本消耗
        indSum = np.sum(ind, axis=0)
        # print(np.sum(indSum, axis=1))
        consumption = np.sum(np.sum(indSum, axis=1) * self.WTAInstance.cost)
        # print(consumption)
        return consumption


if __name__ == '__main__':
    platForm, missile, attacker, pThreshold = 5, 5, 5, 0.95
    instance = WTA(platForm, missile, attacker, pThreshold)
    ind = instance.IndGenerate()
    print(ind)
    fit = instance.GetFitness(ind)
    print(fit)
    tool = Ele(instance)
    ind = tool.IndGenerate()
    print(ind.chromosome)
    print(ind.fitness)
    print(ind.pDamage)
    print(ind.consumption)
