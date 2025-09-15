import matplotlib.pyplot as plt
import numpy as np
import util.utils as utils
from scipy.integrate import odeint
from matplotlib.ticker import ScalarFormatter


# import parameters
from param.params import (
    A_a,a_a,A_gs,a_gs,A_gf,a_gf,A_w,a_w,
    C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,
    phi_e1,phi_e2,v0_p2
)

def f(t, y):
    (y1,y6,y2,y7,y3,y8,y4,y9,y5,y10,y11,y12) = y

    # the model equations
    # P1
    f1 = y6
    f2 = a_a * A_a * utils.sigma(C1 * y2 + C2 * y3 + C3 * (A_a / a_a) * phi_e1 + C11 * y4
                        ) - 2 * a_a * y6 - pow(a_a, 2) * y1
    # SS
    f3 = y7
    f4 = a_a * A_a * utils.sigma(C4 * y1) - 2 * a_a * y7 - pow(a_a, 2) * y2
    # SST (slow inhibitory)
    f5 = y8
    f6 = a_gs * A_gs * utils.sigma(C5 * y1) - 2 * a_gs * y8 - pow(a_gs, 2) * y3
    # P2
    f7 = y9
    f8 = a_a * A_a * utils.sigma(C6 * y4 + C7 * y5 + C8 * (A_a / a_a) * phi_e2 + C12 * y1 + y11,
                           v0=v0_p2) - 2 * a_a * y9 - pow(a_a, 2) * y4
    # PV (fast inhibitory)
    f9 = y10
    f10 = a_gf * A_gf * utils.sigma(C9 * y4 + C10 * y5 + C13 * y1) - 2 * a_gf * y10 - pow(a_gf, 2) * y5
    # whisker stimulation
    f11 = y12
    f12 = a_w * A_w * utils.sigma(utils.whisker(t)) - 2 * a_w * y12 - pow(a_w, 2) * y11

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]

def solve_stepwise(f, Y_init):
    t1 = np.linspace(0, 15, 1500000)
    t2 = np.linspace(15, 30,2500000)

    sol1 =  odeint(f, Y_init, t1, tfirst=True)

    Y_init_2 = sol1[-1]
    sol2 =  odeint(f, Y_init_2, t2, tfirst=True)

    t_combined = np.hstack((t1, t2))
    sol_combined = np.vstack((sol1, sol2))
    return t_combined,sol_combined


# -----------------
# Run simulation
# -----------------
Y_init = np.zeros(12)  # initial condition vector

t_combined, sol_combined = solve_stepwise(f, Y_init)
t_ms = [x * 1000 - 15000 for x in t_combined] # convert to ms


# -----------------
# Process outputs
# -----------------
y_titles = ["y1","y6","y2","y7","y3","y8","y4","y9","y5","y10","y11","y12"]
# remove transient for each
times_list, y_list = [], []
for i in range(12):
    t_i, y_i = utils.process_signal(t_ms, sol_combined[:, i])
    times_list.append(t_i)
    y_list.append(y_i)


# populations & LFP
P1, P2, SS, SST, PV = utils.get_population_activities_with_whisker(sol_combined, C1, C2, C4, C5, C6, C7, C9, C10, C11,  C12, C13)
LFP = utils.get_LFP(P1, P2)

times_LFP, LFP_plot = utils.process_signal(t_ms, LFP)
times_P1, P1_plot   = utils.process_signal(t_ms, P1)
times_P2, P2_plot   = utils.process_signal(t_ms, P2)
times_SS, SS_plot   = utils.process_signal(t_ms, SS)
times_SST, SST_plot = utils.process_signal(t_ms, SST)
times_PV, PV_plot   = utils.process_signal(t_ms, PV)


pop_names = ["P1","P2","SS","SST","PV"]
pop_colors = ["orangered","turquoise","yellowgreen","darkgreen","salmon"]

# Pack each population’s (time, values) into a tuple
pop_series = [
    (times_P1, P1_plot),
    (times_P2, P2_plot),
    (times_PV, PV_plot),
    (times_SS, SS_plot),
    (times_SST, SST_plot)
]


# normalise LFP
max_LFP = max(max(LFP_plot), abs(min(LFP_plot)))
norm_LFP = [x/max_LFP for x in LFP_plot]

# normalise data
mean_lfp_data = np.loadtxt('data/whisker/mean_lfp.csv', delimiter=',', dtype=float) 
t_mean_lfp = np.loadtxt('data/whisker/timestamps.csv', delimiter=',', dtype=float) 

max_data = max(max(mean_lfp_data), abs(min(mean_lfp_data)))
norm_data = [x/max_data for x in mean_lfp_data]


# -----------------
# Plots
# -----------------
# Y variables
plt.figure(figsize=(10,12))
for i,(t_i,y_i) in enumerate(zip(times_list,y_list)):
    plt.subplot(6,2,i+1)
    plt.plot(t_i,y_i,color="darkblue")
    plt.xlim(-10,80)
    plt.xlabel("t (ms)")
    plt.ylabel("V (mV)")
    plt.title(y_titles[i],fontsize=20)
plt.tight_layout()
plt.show()

# LFP proxy
plt.figure(figsize=(8,5))
plt.plot(times_LFP,LFP_plot,color="darkblue",lw=2)
plt.title("LFP proxy",fontsize=20)
plt.xlabel("t (ms)")
plt.ylabel("V (mV)")
plt.xlim(-10,80)
plt.show()

# Populations
plt.figure(figsize=(16,10))
for idx,(name,color,(t_i,vals)) in enumerate(zip(pop_names,pop_colors,pop_series)):
    plt.subplot(3,2,idx+1)
    plt.plot(t_i,vals-vals[0],color=color,lw=2)
    plt.title(name,fontsize=20)
    plt.xlabel("Time (ms)",fontsize=17)
    plt.ylabel("Amplitude (mV)",fontsize=17)
    plt.xlim(-10,80)
    ax=plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
plt.tight_layout()
plt.show()

# Normalised LFP vs data
plt.plot(times_LFP,norm_LFP,label="Normalised LFP proxy",color="darkblue")
plt.plot(t_mean_lfp,norm_data,label="Normalised mean LFP data",color="red")
plt.xlabel("Time (ms)",fontsize=17); plt.ylabel("Amplitude (mV)",fontsize=17)
plt.xlim(-10,80)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
plt.tight_layout()
plt.show()

# LFP vs data (raw)
plt.plot(times_LFP,LFP_plot,label="LFP proxy",color="darkblue")
plt.plot(t_mean_lfp,mean_lfp_data,label="Mean LFP data",color="red")
plt.xlabel("Time (ms)",fontsize=17); plt.ylabel("Amplitude (mV)",fontsize=17)
plt.xlim(-10,80)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
plt.tight_layout()
plt.show()

#############
y1 = y_i[0]
y2 = y_i[2]
y3 = y_i[4]
y4 = y_i[6]
y5 = y_i[8]
y11 = y_i[10]

t1 = t_i[0]
t2 = t_i[2]
t3 = t_i[4]
t4 = t_i[6]
t5 = t_i[8]
t11 = t_i[10]

plt.figure(figsize = (16,10))
plt.subplot(3,2,1)
plt.plot(times_LFP, LFP_plot, color = "darkblue", linewidth = 3)
plt.gca().set_title("LFP proxy = P1 + P2", fontsize = 20)
plt.xlabel('t (ms)', fontsize = 13)
plt.ylabel('V (mV)', fontsize = 13)
plt.xlim(-10,80)

plt.subplot(3,2,3)
plt.plot(times_P1, P1_plot, color = "orangered", label = "P1", linewidth = 3)
plt.plot(t2, C1 * y2, color = "black", linestyle = "dotted", label = "C1 * y2")
plt.plot(t3, C2 * y3, color = "dimgray", linestyle = "dashed", label = "C2 * y3")
plt.plot(t4, C11 * y4, color = "darkslategray", linestyle = "dashdot", label = "C11 * y4")
plt.gca().set_title("P1 = C1 * y2 + C2 * y3 + C11 * y4", fontsize = 20)  
plt.xlabel('t (ms)', fontsize = 13)
plt.ylabel('V (mV)', fontsize = 13)
plt.xlim(-10,80)
plt.legend(loc = "lower right")

plt.subplot(3,2,5)
plt.plot(times_P2, P2_plot, color = "turquoise", label = "P2", linewidth = 3)
plt.plot(t4, C6 * y4, color = "black", linestyle = "dotted", label = "C6 * y4")
plt.plot(t5, C7 * y5, color = "dimgray", linestyle = "dashed", label = "C7 * y5")
plt.plot(t1, C12 * y1, color = "darkslategray", linestyle = "dashdot", label = "C12 * y1")
plt.plot(t11, y11, color = "darkslategray", linestyle = "solid", label = "y11")
plt.gca().set_title("P2 = C6 * y4 + C7 * y5 + C12 * y1 + y11", fontsize = 20)      
plt.xlabel('t (ms)', fontsize = 13)
plt.ylabel('V (mV)', fontsize = 13)
plt.xlim(-10,80)
plt.legend(loc = "lower right")

plt.subplot(3,2,2)
plt.plot(times_PV, PV_plot, color = "salmon", label = "PV", linewidth = 3)
plt.plot(t4, C9 * y4, color = "black", linestyle = "dotted", label = "C9 * y4")
plt.plot(t5, C10 * y5, color = "dimgray", linestyle = "dashed", label = "C10 * y5")
plt.plot(t1, C13 * y1, color = "darkslategray", linestyle = "dashdot", label = "C13 * y1")
plt.gca().set_title("PV = C9 * y4 + C10 * y5 + C13 * y1", fontsize = 20)         
plt.xlabel('t (ms)', fontsize = 13)
plt.ylabel('V (mV)', fontsize = 13)
plt.xlim(-10,80)
plt.legend(loc = "lower right")

plt.subplot(3,2,4)
plt.plot(times_SS, SS_plot, color = "yellowgreen", linewidth = 3)
plt.gca().set_title("SS = C4 * y1", fontsize = 20)         
plt.xlabel('t (ms)', fontsize = 13)
plt.ylabel('V (mV)', fontsize = 13)
plt.xlim(-10,80)

plt.subplot(3,2,6)
plt.plot(times_SST, SST_plot, color = "darkgreen", linewidth = 3) 
plt.gca().set_title("SST = C5 * y1", fontsize = 20)          
plt.xlabel('t (ms)', fontsize = 13)
plt.ylabel('V (mV)', fontsize = 13)
plt.xlim(-10,80)
plt.tight_layout()
plt.show()
