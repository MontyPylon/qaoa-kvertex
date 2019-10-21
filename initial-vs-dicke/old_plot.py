import matplotlib.pyplot as plt
import pickle

dicke_complete = pickle.load(open('data/91.dicke_complete', 'rb'))
dicke_ring = pickle.load(open('data/91.dicke_ring', 'rb'))
init_complete = pickle.load(open('data/91.initial_complete', 'rb'))
init_ring = pickle.load(open('data/91.initial_ring', 'rb'))

plt.plot([x+1 for x in range(len(dicke_complete))], dicke_complete, '-bo', label='Dicke & Complete' )
#plt.plot([x+1 for x in range(len(dicke_ring))], dicke_ring, '-bo', label='Dicke & Ring' )
plt.errorbar([x+1 for x in range(len(init_complete[0]))], init_complete[0], yerr=init_complete[1], fmt='--rv', label='k-state & Complete', capsize=5)
#plt.errorbar([x+1 for x in range(len(init_ring[0]))], init_ring[0], yerr=init_ring[1], fmt='--rv', label='k-state & Ring', capsize=5)

plt.legend(loc=4, fontsize=17)
plt.gca().set_ylabel('Approximation ratio', fontsize=17, labelpad=10)
plt.gca().set_xlabel('$p$', fontsize=17)
plt.xticks([x+1 for x in range(6)], size=17)
plt.yticks([0.8,0.85,0.9,0.95,1], size=17)
plt.gca().set_xlim([0.8,6.2])
#plt.gca().set_ylim([0.89, 1])
#plt.title('$p=$' + str(p))
plt.tight_layout()
plt.show()
