# pylint: disable=import-error

import matplotlib.pyplot as plt
# import statsmodels.tsa.filters.hp_filter as hp
from solver import Solver

fontsize= 14
ticksize = 14
figsize = (12, 4.5)
params = {'font.family':'serif',
    "figure.figsize":figsize,
    'figure.dpi': 80,
    'figure.edgecolor': 'k',
    'font.size': fontsize,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': ticksize,
    'ytick.labelsize': ticksize,
    'text.usetex': True
}
plt.rcParams.update(params)


solver = Solver()

kgrid, kp_pol, c_pol = solver.solve_TI_stoch_ram()

# In the following, we plot the policy functions for consumption and capital (zooming in the last one)

fig,ax=plt.subplots(figsize=(12,7))
ax.plot(kgrid,c_pol[:,0],color="darkblue",linewidth=3,label="$C(K,z_{1})$")

ax.plot(kgrid,c_pol[:,5],color="blue",linewidth=3,label="$C(K,z_{6})$")
ax.plot(kgrid,c_pol[:,10],color="lightblue",linewidth=3,label="$C(K,z_{11})$")

ax.set_ylabel("Consumption $C(K,z)$", fontsize=22)
ax.set_xlabel("Capital Stock K", fontsize=22)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

plt.legend(fontsize= 16, loc='upper left')

plt.show()
fig.savefig("c_pol_rbc.pdf", bbox_inches='tight')



fig,ax=plt.subplots(figsize=(12,7))
ax.plot(kgrid,kp_pol[:,0],color="darkblue",linewidth=3,label="$K'(K,z_{1})$")

ax.plot(kgrid,kp_pol[:,5],color="blue",linewidth=3,label="$K'(K,z_{6})$")
ax.plot(kgrid,kp_pol[:,10],color="lightblue",linewidth=3,label="$K'(K,z_{11})$")

ax.set_ylabel("Capital Choice $K'(K,z)$", fontsize=22)
ax.set_xlabel("Capital Stock K", fontsize=22)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

plt.legend(fontsize= 16, loc='upper left')

plt.show()
fig.savefig("kp_pol_rbc.pdf", bbox_inches='tight')