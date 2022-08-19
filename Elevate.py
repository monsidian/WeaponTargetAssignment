import numpy as np
import copy


class Individual:
    def __init__(self, chromosome, fitness, pDamage, consumption):
        self.chromosome = chromosome
        self.fitness = fitness
        self.pDamage = pDamage
        self.consumption = consumption


class Ele:
    def __init__(self, ammo):
        self.ammo = ammo

    def IndGenerate(self):
        ammo = copy.deepcopy(self.ammo)
        ind = np.zeros(self.dim).astype('int')
        for k in range(self.platForm):
            indPlat = ind[k]
            for i in range(self.missile):
                for j in range(self.attacker):
                    if self.constraint_ptoa[k, j]:
                        indPlat[i, j] = 0
                        continue
                    indPlat[i, j] = np.random.randint(0, ammo[k, i] + 1)
                    ammo[k, i] -= indPlat[i, j]
        dic = self.GetFittness(ind)
        return Individual(ind, dic['fittness'], dic['pDamage'], dic['Consumption'])

    def GetFittness(self, ind):
        # print(self.ammo)
        # print(indSum)
        for k in range(self.platForm):
            for i in range(self.missile):
                if np.sum(ind[k], axis=1)[i] > self.ammo[k, i]:
                    return 0
        pDamage = self.GetpDamage(ind)
        if pDamage < self.pThreshold:
            return pDamage
        consumption = self.GetConsumption(ind)
        fittness = self.pThreshold + np.exp(-consumption / 10000)
        dic = {'fittness': fittness, 'pDamage': pDamage, 'consumption': consumption}
        return dic

    def GetpDamage(self, ind):  # 求毁伤概率
        indSum = np.sum(ind, axis=0)
        p = 1 - np.power(1 - self.p_mtoa, indSum)
        q = 1 - np.prod(1 - p, axis=0)
        # print(q)
        pDamage = np.power(np.cumprod(q, axis=0)[-1], 1 / self.attacker)  # 几何平均
        return pDamage

    def GetConsumption(self, ind):  # 求成本消耗
        indSum = np.sum(ind, axis=0)
        # print(np.sum(indSum, axis=1))
        consumption = np.sum(np.sum(indSum, axis=1) * self.cost)
        # print(consumption)
        return consumption
