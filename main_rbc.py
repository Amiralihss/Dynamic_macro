# pylint: disable=import-error

import matplotlib.pyplot as plt
import statsmodels.tsa.filters.hp_filter as hp
from solver import Solver
from params import Params
from steady_state import Steady_state
from grid_data import Grid_data
import numpy as np
from scipy import interpolate
from agents import *

# TeX Fonts for plots
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
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

kgrid, zgrid, P, kp_pol, n_pol, c_pol, wage, outp, invest = solver.solve_egm_rbc()

# Plot: Policy function for labor supply

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(kgrid, n_pol[:, 0], color="darkblue", linewidth=3, label="$N(K,z_{1})$")
ax.plot(kgrid, n_pol[:, 5], color="blue", linewidth=3, label="$N(K,z_{6})$")
ax.plot(kgrid, n_pol[:, 10], color="lightblue", linewidth=3, label="$N(K,z_{11})$")
ax.set_ylabel("Labor Supply $N(K,z)$", fontsize=22)
ax.set_xlabel("Capital Stock $K$", fontsize=22)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
plt.legend(fontsize=16, loc="upper right")
plt.show()


# Plot: Policy function for consumption

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(kgrid, c_pol[:, 0], color="darkblue", linewidth=3, label="$C(K,z_{1})$")
ax.plot(kgrid, c_pol[:, 5], color="blue", linewidth=3, label="$C(K,z_{6})$")
ax.plot(kgrid, c_pol[:, 10], color="lightblue", linewidth=3, label="$C(K,z_{11})$")
ax.set_ylabel("Consumption $C(K,z)$", fontsize=22)
ax.set_xlabel("Capital Stock $K$", fontsize=22)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
plt.legend(fontsize=16, loc="upper left")
plt.show()

# Plot: Policy function for next period's capital

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(kgrid, kp_pol[:, 0], color="darkblue", linewidth=3, label="$K'(K,z_{1})$")
ax.plot(kgrid, kp_pol[:, 5], color="blue", linewidth=3, label="$K'(K,z_{6})$")
ax.plot(kgrid, kp_pol[:, 10], color="lightblue", linewidth=3, label="$K'(K,z_{11})$")
ax.set_ylabel("Capital Choice $K'(K,z)$", fontsize=22)
ax.set_xlabel("Capital Stock $K$", fontsize=22)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
plt.legend(fontsize=16, loc="upper left")
plt.show()


## PART 2: Compute business cycle moments

# pre-allocation

param = Params(alpha = 0.33,
                beta = 0.99,
                delta = 0.025,
                )
ss = Steady_state(param, labour = True)
# Agents
firm = Firm(labour = True)
household = Household(labour = True)

# Parameters 
alpha = param.alpha
beta = param.beta
delta = param.delta
sigma = param.sigma
gamma = param.gamma
chi = ss.chi

seed = 7122020

np.random.seed(seed)

periods = 2500

c = np.random.uniform(0, 1, periods)
c.shape = (periods, 1)

d = np.zeros((periods, 1), dtype=int)
d[0] = 5


pmat = np.cumsum(P, axis=1)

for i1 in range(1, periods):
    d[i1] = np.argmax(c[i1] < pmat[d[i1 - 1], :])

# Plot: TFP states

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(np.exp(zgrid[d]))
plt.show()


# Simulation

z_sim = zgrid[d]

n_sim = np.zeros(periods)
k_sim = np.zeros(periods + 1)
c_sim = np.zeros(periods)
y_sim = np.zeros(periods)
w_sim = np.zeros(periods)
r_sim = np.zeros(periods)
i_sim = np.zeros(periods)

k_sim[0] = ss.k_ss
print(k_sim[0])

for i1 in range(periods):

    n_interp = interpolate.interp1d(
        kgrid, np.squeeze(n_pol[:, d[i1]]), kind="linear", fill_value="extrapolate"
    )
    n_sim[i1] = n_interp(k_sim[i1])
    r_sim[i1] = (
        firm.mpk(z_sim[i1], k_sim[i1], param, l = n_sim[i1])
    )
    #print(f"z: {z_sim[i1]}, k: {k_sim[i1]}, l: {n_sim[i1]}, alpha: {param.alpha}")
    w_sim[i1] = (
        firm.mpl(z_sim[i1], k_sim[i1], n_sim[i1], param)
    )
    c_sim[i1] = (w_sim[i1] / (chi * n_sim[i1] ** gamma))#**(-1.0/param.sigma)
    y_sim[i1] = firm.f(z_sim[i1], k_sim[i1], param, l = n_sim[i1])
    i_sim[i1] = y_sim[i1] - c_sim[i1]
    k_sim[i1 + 1] = i_sim[i1] + (1 - delta) * k_sim[i1]


# Remove the first 200 periods to avoid dependence on the starting point

burnin = 200

n_ts = np.log(n_sim[burnin:])
k_ts = np.log(k_sim[burnin:])
w_ts = np.log(w_sim[burnin:])
y_ts = np.log(y_sim[burnin:])
i_ts = np.log(i_sim[burnin:])
c_ts = np.log(c_sim[burnin:])
r_ts = r_sim[burnin:]


# Apply the HP filter

lamb = 1600

n_cycle, n_trend = hp.hpfilter(n_ts, lamb)
w_cycle, w_trend = hp.hpfilter(w_ts, lamb)
y_cycle, y_trend = hp.hpfilter(y_ts, lamb)
i_cycle, i_trend = hp.hpfilter(i_ts, lamb)
r_cycle, r_trend = hp.hpfilter(r_ts, lamb)
c_cycle, c_trend = hp.hpfilter(c_ts, lamb)


# Correlation with output

yn_corr = np.corrcoef(n_cycle, y_cycle)[0, 1]
yw_corr = np.corrcoef(w_cycle, y_cycle)[0, 1]
yi_corr = np.corrcoef(i_cycle, y_cycle)[0, 1]
yr_corr = np.corrcoef(r_cycle, y_cycle)[0, 1]
yc_corr = np.corrcoef(c_cycle, y_cycle)[0, 1]


# Autocorrelation

n_autocorr = np.corrcoef(n_cycle[0:-1], n_cycle[1:])[0, 1]
y_autocorr = np.corrcoef(y_cycle[0:-1], y_cycle[1:])[0, 1]
i_autocorr = np.corrcoef(i_cycle[0:-1], i_cycle[1:])[0, 1]
r_autocorr = np.corrcoef(r_cycle[0:-1], r_cycle[1:])[0, 1]
w_autocorr = np.corrcoef(w_cycle[0:-1], w_cycle[1:])[0, 1]
c_autocorr = np.corrcoef(c_cycle[0:-1], c_cycle[1:])[0, 1]


# Relative standard deviation

yn_relstd = np.std(n_cycle) / np.std(y_cycle)
yi_relstd = np.std(i_cycle) / np.std(y_cycle)
yc_relstd = np.std(c_cycle) / np.std(y_cycle)
yr_relstd = np.std(r_cycle) / np.std(y_cycle)
yw_relstd = np.std(w_cycle) / np.std(y_cycle)


# Print the moments

print("")
print("Relative standard deviations")
print("Consumption = " + str(yc_relstd))
print("Investment = " + str(yi_relstd))
print("Hours = " + str(yn_relstd))
print("Wage = " + str(yw_relstd))
print("interest rate = " + str(yr_relstd))
print("")

print("Correlation with output")
print("Consumption = " + str(yc_corr))
print("Investment = " + str(yi_corr))
print("Hours = " + str(yn_corr))
print("Wage = " + str(yw_corr))
print("Interest rate = " + str(yr_corr))
print("")

print("Autocorrelation")
print("Output = " + str(y_autocorr))
print("Consumption = " + str(c_autocorr))
print("Investment = " + str(i_autocorr))
print("Hours = " + str(n_autocorr))
print("Wage = " + str(w_autocorr))
print("Interest rate = " + str(r_autocorr))

