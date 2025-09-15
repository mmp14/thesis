from matplotlib import pyplot as plt
import numpy as np
from param.params import (
    phi_0,r,v0, v0_p2
)

def sigma(v, v0=v0, phi_0=phi_0, r=r):
    return 2 * phi_0 / (1 + np.exp(r * (v0 - v)))


def whisker(t):
    return 20 if 15 <= t < 15.0002 else 0

    
def opto(t, amplitude):
    return amplitude if 15 <= t < 15.02 else 0

def tdcs(t, tdcs_amp):
    if t < 7.5:
        return 0
    elif t < 12.5:
        return (tdcs_amp/5) * (t - 7.5)
    elif t < 17.5:
        return tdcs_amp
    elif t < 22.5:
        return tdcs_amp - (tdcs_amp/5) * (t - 17.5)
    else:
        return 0


def get_population_activities(sol, C1, C2, C4, C5, C6, C7, C9, C10, C11,  C12, C13):
    y1 = sol[0]
    y2 = sol[2]
    y3 = sol[4] 
    y4 = sol[6]
    y5 = sol[8]
    y11 = sol[10]

    P1 = C1 * y2 + C2 * y3 + C11 * y4 
    P2 = C6 * y4 + C7 * y5 + C12 * y1 + y11 
    SS = C4 * y1
    SST = C5 * y1
    PV = C9 * y4 + C10 * y5 + C13 * y1

    return P1, P2, SS, SST, PV

def get_population_activities_with_whisker(sol, C1, C2, C4, C5, C6, C7, C9, C10, C11, C12, C13):
    y1 = sol[:,0]
    y2 = sol[:,2]
    y3 = sol[:,4] 
    y4 = sol[:,6]
    y5 = sol[:,8]
    yw = sol[:,10]

    P1 = C1 * y2 + C2 * y3 + C11 * y4 
    P2 = C6 * y4 + C7 * y5 + C12 * y1 + yw
    SS = C4 * y1
    SST = C5 * y1
    PV = C9 * y4 + C10 * y5 + C13 * y1

    return P1, P2, SS, SST, PV

def get_population_activities_with_whisker_opto_glut(sol, C1, C2, C4, C5, C6, C7, C9, C10, C11, C12, C13):
    y1 = sol[:,0]
    y2 = sol[:,2]
    y3 = sol[:,4] 
    y4 = sol[:,6]
    y5 = sol[:,8]
    yw = sol[:,10]
    yo = sol[:,12]

    P1 = C1 * y2 + C2 * y3 + C11 * y4 + yo
    P2 = C6 * y4 + C7 * y5 + C12 * y1 + yw + yo
    SS = C4 * y1 + yo
    SST = C5 * y1 
    PV = C9 * y4 + C10 * y5 + C13 * y1

    return P1, P2, SS, SST, PV

def get_population_activities_with_whisker_opto_gaba(sol, C1, C2, C4, C5, C6, C7, C9, C10, C11, C12, C13):
    y1 = sol[:,0]
    y2 = sol[:,2]
    y3 = sol[:,4] 
    y4 = sol[:,6]
    y5 = sol[:,8]
    yw = sol[:,10]
    yo = sol[:,12]

    P1 = C1 * y2 + C2 * y3 + C11 * y4
    P2 = C6 * y4 + C7 * y5 + C12 * y1 + yw
    SS = C4 * y1
    SST = C5 * y1 + yo
    PV = C9 * y4 + C10 * y5 + C13 * y1 + yo

    return P1, P2, SS, SST, PV

def get_population_activities_with_opto_gaba(sol, C1, C2, C4, C5, C6, C7, C9, C10, C11, C12, C13):
    y1 = sol[:,0]
    y2 = sol[:,2]
    y3 = sol[:,4] 
    y4 = sol[:,6]
    y5 = sol[:,8]
    yo = sol[:,10]

    P1 = C1 * y2 + C2 * y3 + C11 * y4
    P2 = C6 * y4 + C7 * y5 + C12 * y1
    SS = C4 * y1
    SST = C5 * y1 + yo
    PV = C9 * y4 + C10 * y5 + C13 * y1 + yo

    return P1, P2, SS, SST, PV

def get_population_activities_with_opto_glut(sol, C1, C2, C4, C5, C6, C7, C9, C10, C11, C12, C13):
    y1 = sol[:,0]
    y2 = sol[:,2]
    y3 = sol[:,4] 
    y4 = sol[:,6]
    y5 = sol[:,8]
    yo = sol[:,10]

    P1 = C1 * y2 + C2 * y3 + C11 * y4 + yo
    P2 = C6 * y4 + C7 * y5 + C12 * y1 + yo
    SS = C4 * y1 + yo
    SST = C5 * y1
    PV = C9 * y4 + C10 * y5 + C13 * y1

    return P1, P2, SS, SST, PV

def get_LFP(P1, P2):
    LFP = -(P1 + P2)
    return LFP

def process_signal(t_ms, signal):
    times, steady = remove_transient_plot(t_ms, signal)
    return times, steady - steady[0]


def remove_transient_plot(times, signal):
    times = np.array(times)

    idx_10ms_before_whisker = np.where(times >= -10)[0][0]

    times_plot = times[idx_10ms_before_whisker:len(times)]
    signal_plot = signal[idx_10ms_before_whisker:len(signal)]

    return times_plot, signal_plot


def process_signals(t_ms, P1, P2, SS, SST, PV, LFP):
    times, LFP_steady = process_signal(t_ms, LFP)
    times_P1, P1_steady  = process_signal(t_ms, P1)
    times_P2, P2_steady  = process_signal(t_ms, P2)
    times_SS, SS_steady  = process_signal(t_ms, SS)
    times_SST, SST_steady = process_signal(t_ms, SST)
    times_PV, PV_steady  = process_signal(t_ms, PV)
    return times,LFP_steady,P1_steady,P2_steady,SS_steady,SST_steady,PV_steady

def populate_potential_array(pop_potentials, P1_steady, P2_steady, SS_steady, SST_steady, PV_steady):
    pop_potentials[0].append(P1_steady)
    pop_potentials[1].append(P2_steady)
    pop_potentials[2].append(SS_steady)
    pop_potentials[3].append(SST_steady)
    pop_potentials[4].append(PV_steady)

def populate_firing_rate_array(pop_rates, P1_steady, P2_steady, SS_steady, SST_steady, PV_steady):
    pop_rates[0].append([sigma(x)            for x in P1_steady])
    pop_rates[1].append([sigma(x, v0=v0_p2) for x in P2_steady])
    pop_rates[2].append([sigma(x)            for x in SS_steady])
    pop_rates[3].append([sigma(x)           for x in SST_steady])
    pop_rates[4].append([sigma(x)            for x in PV_steady])

def plot_LFPs(amplitudes, LFP_array, times_LFP, colour):
    plt.figure(figsize = (8,7.5))
    for i in range(len(amplitudes)):
        if i == 0:
            plt.plot(times_LFP, LFP_array[i], color = colour[i], linestyle = "dashed", label = "No photostim.") 
        else:
            plt.plot(times_LFP, LFP_array[i], color = colour[i], label = "Photostim amplitude = " + str(amplitudes[i])) 
# Initialise empty lists to store legend handles and labels
    handles, labels = None, None
    if handles is None and labels is None:
        handles, labels = plt.gca().get_legend_handles_labels()
    plt.xlabel('Time (ms)', fontsize = 25)
    plt.ylabel('Amplitude (mV)', fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
# plt.gca().set_title("LFP proxy", fontsize = 20)
    plt.xlim(-10,40)
    ax=plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# Make the offset text larger
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_fontsize(20) 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
           ncol=1, fontsize=20, frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35) 
    plt.show()
    return handles,labels

def plot_y(amplitudes, y_arrays, t_arrays, colour):
    y_titles = ["y1","y6","y2","y7","y3","y8","y4","y9","y5","y10","y11","y12"]
    plt.figure(figsize=(10,15))
    for j in range(12):
        plt.subplot(7,2,j+1)
        for i in range(len(amplitudes)):
            plt.plot(t_arrays[j][i],y_arrays[j][i],color=colour[i])
        plt.title(y_titles[j])
        plt.xlim(-10,80)
        plt.xlabel('t (ms)', fontsize = 13)
        plt.ylabel('V (mV)', fontsize = 13)
    plt.tight_layout()
    plt.show()

def plot_potentials(amplitudes, pop_potential, time, colour, handles, labels):
    titles = ["P1","P2","PV","SS","SST"]
    plt.figure(figsize = (14,12))
    for j in range(5):
        plt.subplot(3,2,j+1)
        for i in range(len(amplitudes)):
            plt.plot(time,pop_potential[j][i],color=colour[i])
        plt.title(titles[j])
        plt.xlim(-10,80)
        plt.xlabel('t (ms)', fontsize = 13)
        plt.ylabel('V (mV)', fontsize = 13)
        plt.figlegend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.000001), fontsize = 13)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Adjust layout to make space for legend
    plt.show()


def plot_firing_rates(amplitudes, pop_rates, times, colour, handles, labels):
    titles = ["P1","P2","PV","SS","SST"]
    plt.figure(figsize = (14,12))
    for j in range(5):
        plt.subplot(3,2,j+1)
        for i in range(len(amplitudes)):
            plt.plot(times, pop_rates[j][i],color=colour[i])
        plt.title(titles[j])
        plt.xlim(-10,80)
        plt.xlabel('t (ms)', fontsize = 13)
        plt.ylabel('Firing rate (Hz)', fontsize = 13)
        plt.figlegend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.000001), fontsize = 13)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Adjust layout to make space for legend
    plt.show()