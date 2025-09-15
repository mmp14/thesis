from scipy.integrate import odeint
import numpy as np
import util.utils as utils

# import parameters
from param.params import (
    A_a,a_a,A_gs,a_gs,A_gf,a_gf,A_o,a_o,
    C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,
    phi_e1,phi_e2,v0_p2
)

opto_amp = .5
tdcs_amplitudes = [-.75, -.5,  0, .5, .75]

def f(t, y, opto_amplitude, tdcs_amp):
    (y1,y6,y2,y7,y3,y8,y4,y9,y5,y10,y11,y12) = y
    # the model equations
    #P1
    f1 = y6
    f2 = a_a * A_a * utils.sigma(C1 * y2 + C2 * y3 + C3 * (A_a / a_a) * phi_e1 + C11 * y4 + utils.tdcs(t, tdcs_amp),
                           ) - 2 * a_a * y6 - pow(a_a, 2) * y1
    #SS 
    f3 = y7
    f4 = a_a * A_a * utils.sigma(C4 * y1) - 2 * a_a * y7 - pow(a_a, 2) * y2
    #SST (slow inhibitory)
    f5 = y8
    f6 = a_gs * A_gs * utils.sigma(C5 * y1 + y11) - 2 * a_gs * y8 - pow(a_gs, 2) * y3
    #P2
    f7 = y9
    f8 = a_a * A_a * utils.sigma(C6 * y4 + C7 * y5 + C8 * (A_a / a_a) * (phi_e2) + C12 * y1 + utils.tdcs(t, tdcs_amp),
                           v0=v0_p2) - 2 * a_a * y9 - pow(a_a, 2) * y4
    #PV (fast inhibitory)
    f9 = y10
    f10 = a_gf * A_gf * utils.sigma(C9 * y4 + C10 * y5 + C13 * y1 + y11) - 2 * a_gf * y10 - pow(a_gf, 2) * y5
    #opto
    f11 = y12
    f12 = a_o * A_o * utils.sigma(utils.opto(t, opto_amplitude)) - 2 * a_o * y12 - pow(a_o, 2) * y11
    
    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]


def solve_stepwise(f, Y_init, opto_amplitude, tdcs_amplitude):
    args = (opto_amplitude, tdcs_amplitude,)

    t1 = np.linspace(0, 15, 1500000)
    t2 = np.linspace(15, 30,2500000)

    sol1 =  odeint(f, Y_init, t1, args = args, tfirst=True)

    Y_init_2 = sol1[-1]
    sol2 =  odeint(f, Y_init_2, t2, args = args, tfirst=True)

    t_combined = np.hstack((t1, t2))
    sol_combined = np.vstack((sol1, sol2))
    return t_combined,sol_combined

Y_init = np.zeros(12)  # initial condition vector

LFP_array= []
pop_potentials = [[] for i in range(5)] # one list per pop
pop_rates = [[] for i in range(5)] # one list per pop
y_arrays = [[] for _ in range(12)]  # one list per state variab
t_arrays = [[] for _ in range(12)]  # one list per state variab

for tdcs_amp  in tdcs_amplitudes:

    t_combined, sol_combined = solve_stepwise(f, Y_init, opto_amp, tdcs_amp)
    t_ms = [x * 1000 - 15000 for x in t_combined]

    for i in range(12):
        times_i, y_i = utils.process_signal(t_ms, sol_combined[:, i])
        t_arrays[i].append(times_i)
        y_arrays[i].append(y_i)

    P1, P2, SS, SST, PV = utils.get_population_activities_with_opto_gaba(sol_combined, C1, C2, C4, C5, C6, C7, C9, C10, C11, C12, C13)

    LFP = utils.get_LFP(P1, P2)

    times, LFP_steady, P1_steady, P2_steady, SS_steady, SST_steady, PV_steady = utils.process_signals(t_ms, P1, P2, SS, SST, PV, LFP)

    LFP_array.append(LFP_steady)

    # potentials
    utils.populate_potential_array(pop_potentials, P1_steady, P2_steady, SS_steady, SST_steady, PV_steady)

    # firing rates
    utils.populate_firing_rate_array(pop_rates, P1_steady, P2_steady, SS_steady, SST_steady, PV_steady)

########
# Plot #
#########
colour = ["xkcd:dark blue", "xkcd:blue", "xkcd:grey", "xkcd:salmon", "xkcd:bright red"]

handles, labels = utils.plot_LFPs(tdcs_amplitudes, LFP_array, times, colour)

utils.plot_potentials(tdcs_amplitudes, pop_potentials, times, colour, handles, labels)

utils.plot_firing_rates(tdcs_amplitudes, pop_rates, times, colour, handles, labels)

utils.plot_y(tdcs_amplitudes, y_arrays, t_arrays, colour)
