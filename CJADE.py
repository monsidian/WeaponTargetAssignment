import numpy as np
import copy
import time
from WeaponTargetAssignment import Ele


class Cjade:
    def __init__(self, iterNum, popNum, instance, pMutation=0.2, uF=0.5, uCR=0.1, c=0.5):
        self.name = 'CJADE'
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

        self.lastNums = self.integertoDecimal(self.bestInd)

    def logisticTentSystem(self, nums, u=2):
        return np.where(nums < 0.5, u * nums * (1 - nums) + (4 - u) / 2 * nums,
                        u * nums * (1 - nums) + (4 - u) / 2 * (1 - nums))

    def integertoDecimal(self, ind):
        ammo = copy.deepcopy(self.instance.ammo)
        nums = np.zeros(self.instance.dim)
        for i in range(self.instance.attacker):
            nums[:, :, i] = ind.chromosome[:, :, i] / ammo
            ammo -= ind.chromosome[:, :, i]
        return nums

    def decimaltoInteger(self, nums):
        ammo = copy.deepcopy(self.instance.ammo)
        chromosome = np.zeros(self.instance.dim)
        for i in range(self.instance.attacker):
            chromosome[:, :, i] = nums[:, :, i] * ammo
            chromosome = chromosome.astype('int')
            ammo -= chromosome[:, :, i]
        return chromosome

    def reConstruct(self, u, lastBest):
        uChaos = copy.deepcopy(u)
        if lastBest != uChaos[0]:
            nums = self.integertoDecimal(uChaos[0])
        else:
            nums = self.logisticTentSystem(self.lastNums)
        uNums = [nums]
        for i in range(1, self.popNum):
            tmpNums = self.logisticTentSystem(uNums[i - 1])
            uNums.append(tmpNums)
            uChaos[i].chromosome = self.decimaltoInteger(tmpNums)
            uChaos[i].chromosome = np.where(uChaos[i].chromosome < 0, 0, uChaos[i].chromosome)
            dic = self.tool.GetFitness(uChaos[i].chromosome)
            uChaos[i].fitness, uChaos[i].pDamage, uChaos[i].consumption = dic['fitness'], dic['pDamage'], dic[
                'consumption']
        self.lastNums = uNums[-1]
        uChaos.sort(key=lambda x: x.fitness, reverse=True)
        return uChaos



    def mutation(self, u):
        uMutant = copy.deepcopy(u)
        for i, indMutant in enumerate(uMutant):
            indMutant.F = np.random.normal(self.uF, 0.1)
            indBest = np.random.choice(u[:int(self.pMutation * self.popNum)])
            indR1 = np.random.choice(u)
            indR2 = np.random.choice(u + self.archive)
            indMutant.chromosome = (indMutant.chromosome + indMutant.F * (
                        indBest.chromosome - indMutant.chromosome) + indMutant.F * (
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
            np.where(indCross.chromosome < 0, 0, indCross.chromosome)
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
        localCont = 0
        lastBest = self.bestInd
        for i in range(self.iterNum):
            uMutant = self.mutation(u)
            uCross = self.crossOver(uMutant)
            u = self.select(u, uCross)

            self.bestInd = u[0]
            self.iterF.append(self.bestInd.fitness)
            self.iterPDamage.append(self.bestInd.pDamage)
            self.iterConsumption.append(self.bestInd.consumption)

            if self.bestInd.fitness > lastBest.fitness:
                lastBest = self.bestInd
            else:
                localCont += 1

            if localCont >= 50:
                self.archive += u[1:]
                u = self.reConstruct(u, lastBest)
                localCont = 0

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
