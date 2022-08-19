import numpy as np
import copy
from Elevate import Ele


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

    # def IndGenerate(self):
    #     ammo = copy.deepcopy(self.ammo)
    #     ind = np.zeros(self.dim).astype('int')
    #     for k in range(self.platForm):
    #         indPlat = ind[k]
    #         for i in range(self.missile):
    #             for j in range(self.attacker):
    #                 if self.constraint_ptoa[k, j]:
    #                     indPlat[i, j] = 0
    #                     continue
    #                 indPlat[i, j] = np.random.randint(0, ammo[k, i] + 1)
    #                 ammo[k, i] -= indPlat[i, j]
    #     return ind
    #
    # def GetFittness(self, ind):
    #     # print(self.ammo)
    #     # print(indSum)
    #     for k in range(self.platForm):
    #         for i in range(self.missile):
    #             if np.sum(ind[k], axis=1)[i] > self.ammo[k, i]:
    #                 return 0
    #     pDamage = self.GetpDamage(ind)
    #     if pDamage < self.pThreshold:
    #         return pDamage
    #     consumption = self.GetConsumption(ind)
    #     fittness = self.pThreshold + np.exp(-consumption / 10000)
    #     return fittness
    #
    # def GetpDamage(self, ind):  # 求毁伤概率
    #     indSum = np.sum(ind, axis=0)
    #     p = 1 - np.power(1 - self.p_mtoa, indSum)
    #     q = 1 - np.prod(1 - p, axis=0)
    #     # print(q)
    #     pDamage = np.power(np.cumprod(q, axis=0)[-1], 1 / self.attacker)  # 几何平均
    #     return pDamage
    #
    # def GetConsumption(self, ind):  # 求成本消耗
    #     indSum = np.sum(ind, axis=0)
    #     # print(np.sum(indSum, axis=1))
    #     consumption = np.sum(np.sum(indSum, axis=1) * self.cost)
    #     # print(consumption)
    #     return consumption

platForm, missile, attacker = 5, 5, 5
pThreshold = 0.95
instance = WTA(platForm, missile, attacker, pThreshold)
# ind = instance.IndGenerate()
# print(ind)
# fit = instance.GetFittness(ind)
# print(fit)

tool = Ele(instance.ammo)
ind = tool.IndGenerate()
print(ind)