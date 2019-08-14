import matplotlib.pyplot as plt
import pickle

monte = pickle.load(open('data/samples', 'rb'))

plt.errorbar([x+1 for x in range(len(monte[0]))], monte[0], yerr=monte[1], fmt='--ro', capsize=5)

#plt.legend(loc=4, fontsize=17)
plt.gca().set_ylabel('Number of samples', fontsize=17, labelpad=10)
plt.gca().set_xlabel('$p$', fontsize=17)
plt.xticks([x+1 for x in range(6)], size=17)
#plt.yticks([0.8,0.85,0.9,0.95,1], size=17)
plt.yticks(size=17)
plt.gca().set_xlim([0.8,6.2])
plt.yscale('log')
#plt.gca().set_ylim([0.89, 1])
#plt.title('$p=$' + str(p))
plt.tight_layout()
plt.show()
