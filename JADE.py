import numpy as np
import copy
import time
from WeaponTargetAssignment import Ele


class Jade:
    def __init__(self, iterNum, popNum, instance, pMutation=0.2, uF=0.5, uCR=0.1, c=0.5):
        self.name = 'JADE'
        self.instance = instance
        self.iterNum = iterNum
        self.popNum = popNum

        # self.Fs = np.zeros(popNum)
        # self.CRs = np.zeros((popNum, instance.platForm, instance.missile))
        self.c = c
        self.uF = uF
        self.uCR = uCR
        self.pMutation = pMutation
        self.SetF = []
        self.SetCR = []
        self.archive = []

        self.platForm, self.missile, self.attacker = instance.dim
        self.constraint_ptoa = instance.constraint_ptoa
        self.tool = Ele(instance)
        self.uOri = []
        for _ in range(popNum):
            ind = self.tool.IndGenerate()
            ind.setCR(instance.platForm, instance.missile)
            self.uOri.append(ind)
        self.uOri.sort(key=lambda x: x.fitness, reverse=True)

        self.bestInd = self.uOri[0]
        self.iterF = [self.bestInd.fitness]
        self.iterPDamage = [self.bestInd.pDamage]
        self.iterConsumption = [self.bestInd.consumption]

    def mutation(self, u):
        uMutant = copy.deepcopy(u)
        for i, indMutant in enumerate(uMutant):
            indMutant.F = np.random.normal(self.uF, 0.1)
            indBest = np.random.choice(u[:int(self.pMutation * self.popNum)])
            indR1 = np.random.choice(u)
            indR2 = np.random.choice(u + self.archive)
            indMutant.chromosome = (indMutant.chromosome + indMutant.F * (indBest.chromosome - indMutant.chromosome) + indMutant.F * (
                        indR1.chromosome - indR2.chromosome)).astype('int')
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
        for h, indCross in enumerate(uCross):
            for k, genPlat in enumerate(indCross.chromosome):
                for i, genWeapon in enumerate(genPlat):
                    indCross.CR[k, i] = np.random.normal(self.uCR, 0.1)
                    if np.random.random() < indCross.CR[k, i]:
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
                self.archive.append(indParent)
            else:
                uNew.append(indParent)
                # print(indParent.fitness == indChild.fitness)
        uNew.sort(key=lambda x: x.fitness, reverse=True)
        while len(self.archive) > self.popNum:
            del self.archive[np.random.randint(0, len(self.archive) - 1)]
        Fs = [ind.F for ind in uNew]
        CRs = [ind.CR for ind in uNew]
        self.uF = (1 - self.c) * self.uF + self.c * self.LehmerAve(Fs)
        self.uCR = (1 - self.c) * self.uCR + self.c * self.average(CRs)
        return uNew
    def average(self, nums):
        return np.mean(nums)
    def LehmerAve(self, nums):
        squareSum = 0
        normalSum = 0
        for num in nums:
            squareSum += num ** 2
            normalSum += num
        return squareSum / normalSum if normalSum else self.uF
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
