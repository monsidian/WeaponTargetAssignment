import numpy as np
import copy
import time
from WeaponTargetAssignment import Ele


class DEPro:
    def __init__(self, iterNum, popNum, instance, F=0.5, CR=0.8):
        self.name = 'DEPro'
        self.instance = instance
        self.iterNum = iterNum
        self.F = F
        self.CR = CR
        self.platForm, self.missile, self.attacker = instance.dim
        self.constraint_ptoa = instance.constraint_ptoa
        self.tool = Ele(instance)
        self.uOri = [self.tool.IndGenerate() for _ in range(popNum)]
        self.uOri.sort(key=lambda x: x.fitness, reverse=True)

        self.bestInd = self.uOri[0]
        self.iterF = [self.bestInd.fitness]
        self.iterPDamage = [self.bestInd.pDamage]
        self.iterConsumption = [self.bestInd.consumption]

    def mutation(self, u):
        uMutant = copy.deepcopy(u)
        for indMutant in uMutant:
            indR1, indR2, indR3 = np.random.choice(u, 3, replace=False)
            indMutant.chromosome = (indR1.chromosome + self.F * (indR2.chromosome - indR3.chromosome)).astype('int')
            index = np.where(indMutant.chromosome < 0)
            indMutant.chromosome[index] = 0
            dic = self.tool.GetFitness(indMutant.chromosome)
            indMutant.fitness, indMutant.pDamage, indMutant.consumption = dic['fitness'], dic['pDamage'], dic[
                'consumption']
            # print(indMutant.fitness)
        return uMutant

    def crossOver(self, uMutant):
        # print(uMutant)
        uCross = copy.deepcopy(uMutant)
        ammo = copy.deepcopy(self.instance.ammo)
        for indCross in uCross:
            if np.random.random() < self.CR:
                index = np.argwhere(self.constraint_ptoa == 0)
                k, j = index[np.random.choice(len(index))]
                i = np.random.randint(0, self.missile)
                indCross.chromosome[k, i, j] = np.random.randint(0, ammo[k, i] + 1)
                ammo[k, i] -= indCross.chromosome[k, i, j]
                # for k, genPlat in enumerate(indCross.chromosome):
                #     for i, genWeapon in enumerate(genPlat):
                #         for j, decisionVariable in enumerate(genWeapon):
                #             if self.instance.constraint_ptoa[k, j]:
                #                 genWeapon[j] = 0
                #             else:
                #                 genWeapon[j] = np.random.randint(0, ammo[k, i] + 1)
                #                 ammo[k, i] -= genWeapon[j]
            index = np.where(indCross.chromosome < 0)
            indCross.chromosome[index] = 0
            dic = self.tool.GetFitness(indCross.chromosome)
            indCross.fitness, indCross.pDamage, indCross.consumption = dic['fitness'], dic['pDamage'], dic[
                'consumption']
        return uCross

    def crossOverOri(self, uMutant):
        # print(uMutant)
        uCross = copy.deepcopy(uMutant)
        ammo = copy.deepcopy(self.instance.ammo)
        for indCross in uCross:
            for k, genPlat in enumerate(indCross.chromosome):
                for i, genWeapon in enumerate(genPlat):
                    if np.random.random() < self.CR:
                        for j, decisionVariable in enumerate(genWeapon):
                            if self.instance.constraint_ptoa[k, j]:
                                genWeapon[j] = 0
                            else:
                                genWeapon[j] = np.random.randint(0, ammo[k, i] + 1)
                                ammo[k, i] -= genWeapon[j]
            index = np.where(indCross.chromosome < 0)
            indCross.chromosome[index] = 0
            dic = self.tool.GetFitness(indCross.chromosome)
            indCross.fitness, indCross.pDamage, indCross.consumption = dic['fitness'], dic['pDamage'], dic[
                'consumption']
        return uCross


    def select(self, uParent, uChild):
        uNew = []
        for indParent, indChild in zip(uParent, uChild):
            # print(indParent.fitness, indChild.fitness)
            if indParent.fitness < indChild.fitness:
                uNew.append(indChild)
            else:
                uNew.append(indParent)
                # print(indParent.fitness == indChild.fitness)
        uNew.sort(key=lambda x: x.fitness, reverse=True)
        return uNew

    def update(self):
        timeStart = time.time()
        u = self.uOri
        for i in range(self.iterNum):
            uMutant = self.mutation(u)
            uCross = self.crossOver(uMutant)
            u = self.select(u, uCross)

            self.bestInd = u[0]
            self.iterF.append(self.bestInd.fitness)
            self.iterPDamage.append(self.bestInd.pDamage)
            self.iterConsumption.append(self.bestInd.consumption)

            if not i % 100:
                print(i, self.bestInd.consumption)
        timeEnd = time.time()
        timeSpend = timeEnd - timeStart
        dic = {'bestInd': self.bestInd,
               'iterF': self.iterF,
               'iterPDamage': self.iterPDamage,
               'iterConsumption': self.iterConsumption,
               'time': timeSpend}
        return dic


