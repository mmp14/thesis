# Laminar Neural Mass Model validation with optogenetic tDCS data

This repository contains the simulation code for a master's thesis investigating the effects of combined whisker stimulation, transcranial direct current stimulation (tDCS), and optogenetics on local field potentials (LFP) using a laminar neural mass model.

The model used is a combination of the Jansen-Rit and PING models and is described in Sanchez-Todo, R. et al. A physical neural mass model framework for the analy- sis of oscillatory generators from laminar electrophysiological recordings. (2023)

## Overview

The model simulates cortical laminar dynamics in the whisker barrel cortex under several experimental conditions:

- **Whisker stimulation alone** — baseline sensory-evoked LFP responses
- **Optogenetics only** — GABAergic or glutamatergic optogenetic drive without sensory input
- **tDCS + whisker stimulation** — effect of DC current on sensory-evoked responses
- **tDCS + optogenetics** — combined neuromodulation without sensory drive
- **Combined protocols** — whisker stimulation with optogenetic modulation under tDCS

Each condition is modelled separately for **GABAergic** and **glutamatergic** optogenetic targets, reflecting distinct inhibitory and excitatory circuit effects.

## Repository Structure

```
thesis/
├── simulations/
│   ├── whisker/
│   │   ├── whisker.py                    # Whisker stimulation only
│   ├── whisker_opto/
│   │   ├── opto_gaba.py                  # Whisker + GABAergic optogenetics
│   │   └── opto_glut.py                  # Whisker + glutamatergic optogenetics
│   └── whisker_tdcs/
│       ├── tdcs_whisker.py               # tDCS + whisker stimulation
└── opto_only/
│       ├──opto_gaba.py                   # Optogenetics only (GABA)
│       ├──opto_glut.py                   # Optogenetics only (Glut)
└── tdcs_opto/
│       ├── tdcs_opto_gaba.py            # tDCS + GABAergic optogenetics only
│       └── tdcs_opto_glut.py            # tDCS + Glutamatergic optogenetics only
├── param/                                # Model parameters
└── util/                                 # Utility functions
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib (for plotting results)

Install dependencies with:

```bash
pip install numpy scipy matplotlib
```

## Usage

Each script is self-contained and can be run directly. For example, to simulate whisker stimulation alone:

```bash
python whisker.py
```

To simulate the effect of tDCS combined with whisker stimulation:

```bash
python tdcs_whisker_new_params.py
```

Scripts prefixed with `tdcs_` apply a tDCS current to the corresponding baseline condition. Scripts suffixed with `_gaba_` or `_glut_` specify the optogenetic target population.

### Parameter Configuration

Model parameters (e.g. synaptic weights, time constants, stimulation amplitudes) are defined in the `param/` directory. Modify these files to explore different parameter regimes without editing the main simulation scripts.

### Utility Functions

Shared helper functions (e.g. LFP computation, signal filtering, plotting) are located in the `util/` directory and imported by the main scripts.

## Model Description

The model is a **laminar neural mass model** representing distinct cortical layers. Each layer contains excitatory and inhibitory neural populations whose mean firing rates are coupled via synaptic connectivity matrices. The LFP is computed as a weighted sum of synaptic currents across layers.

Neuromodulatory inputs are incorporated as follows:

- **Whisker stimulation** — modelled as a time-varying thalamic input to layer IV
- **tDCS** — implemented as a shift in neuronal excitability (resting membrane potential offset) proportional to the applied current density
- **Optogenetics** — modelled as an additional excitatory (glutamatergic) or inhibitory (GABAergic) drive targeting specific populations


## License

This code is made available for academic and research purposes. Please contact the author before reusing or adapting it for other projects.
