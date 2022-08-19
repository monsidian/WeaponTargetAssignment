from matplotlib import pyplot as plt


def visualize(dics, pThreshold):
    plt.figure(1)
    plt.axhline(pThreshold, color='black', label='ThresholdProbability')
    for algorithm, dic in dics.items():
        iterPDamage = dic['iterPDamage']
        iterConsumption = dic['iterConsumption']
        iterFitness = dic['iterF']
        time = dic['time']
        # print(algorithm + '分配方案为：')
        # print(dic['bestInd'].chromosome)
        print(algorithm + '平均毁伤概率为：' + str(iterPDamage[-1]))
        print(algorithm + '成本消耗为：' + str(iterConsumption[-1]))
        print(algorithm + '耗时为：' + str(time) + 's')

        plt.figure(1)
        plt.plot(iterPDamage, label=algorithm)
        plt.xlabel('iteration')
        plt.ylabel('DamageProbability')
        plt.legend(loc='best')

        plt.figure(2)
        plt.plot(iterConsumption, label=algorithm)
        plt.xlabel('iteration')
        plt.ylabel('Consumption')
        plt.legend(loc='best')

        plt.figure(3)
        plt.plot(iterFitness, label=algorithm)
        plt.xlabel('iteration')
        plt.ylabel('Fitness')
        plt.legend(loc='best')
    plt.show()

