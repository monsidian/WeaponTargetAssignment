import numpy as np
from WeaponTargetAssignment import WTA
from DifferentialEvolution import DE
from DifferentialEvolutionPro import DEPro
from JADE import Jade
from JADEPro import JadePro
from CJADE import Cjade
from CJADEPro import CjadePro
from Visualization import visualize

platForm, missile, attacker, pThreshold = 6,6,6, 0.95
# platForm = int(input('请输入平台数：\n'))
# missile = int(input('请输入武器数：\n'))
# attacker = int(input('请输入来袭目标数：\n'))
instance = WTA(platForm, missile, attacker, pThreshold)

algorithms = [Cjade, CjadePro, DE, DEPro, Jade, JadePro]
algorithms = [Cjade, CjadePro, DE, Jade]
# algorithms = ['DE', 'DEPro']
dics = {}
iterNum, popNum = 500, 10
testNum = 50
for algorithm in algorithms:
    al = algorithm(iterNum, popNum, instance)
    print(al.name + ':')
    dics[al.name] = al.update()

visualize(dics, pThreshold)
