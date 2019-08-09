import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

total_c = []
total_r = []
for gi in range(163, 955):
    path = 'complete/' + str(gi) + '.mpi'
    data = []
    if os.path.exists(path):
        with open(path, 'rb') as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                print(e)
        if data: total_c.append(data)

    path = 'ring/' + str(gi) + '.mpi'
    data = []
    if os.path.exists(path):
        with open(path, 'rb') as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                print(e)
        if data: total_r.append(data)


ratio = []
for i in range(len(total_c[0])):
    local = []
    for j in range(len(total_c)):
        local.append(total_c[j][i]/total_r[j][i])
    ratio.append(local)

avg, std = [], []
for i in range(len(total_c[0])):
    avg.append(np.average(ratio[i]))
    std.append(np.std(ratio[i]))

error = [i*2.576/np.sqrt(len(total_c)) for i in std]

x = [i+1 for i in range(6)]

plt.errorbar(x, avg, yerr=error, fmt='-o', color='k', capsize=7)
#plt.legend(loc=4, fontsize=17)
plt.gca().set_ylabel('$\\frac{r_K}{r_R}$', rotation=0, fontsize=25, labelpad=30)
plt.gca().set_xlabel('$p$', fontsize=17)
plt.xticks([x+1 for x in range(6)], size=17)
plt.yticks([1.0, 1.02, 1.04, 1.06, 1.08, 1.1], size=17)
plt.gca().set_xlim([0.8,6.2])
plt.gca().set_ylim([1, 1.10])
#plt.title('$p=$' + str(p))
plt.tight_layout()
plt.show()

