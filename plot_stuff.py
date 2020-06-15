import k2model

N=10**4
realisations=1000
copy_prob = np.zeros(N)
L_prob = np.zeros(N)
for i in range(realisations):
    G = k2model.PA_graph(n=2)
    G.add_nodes(N)
    copy_prob += np.array(G.copy_prob)
    L_prob += np.array(G.L_prob)
copy_prob /= float(realisations)
L_prob /= float(realisations)
plt.figure()
plt.plot(copy_prob,'.',label='copy_prob')
plt.plot(L_prob,'.',label='L_prob')
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.show()
