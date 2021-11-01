# Custom libraries
import alabujos as ala

# Public libraries
import pandas as pd
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns

sns.set()

def bold(string):
    chr_start = "\033[1m"
    chr_end = "\033[0m"
    print(chr_start + string + chr_end)


def underline(string):
    chr_start = "\033[4m"
    chr_end = "\033[0m"
    print(chr_start + string + chr_end)


def color_lib(categories):
    """
    This function generates a dictionary that assigns a specific color from a color spectrum to each fund.
    It enables consistent coloring across visualizations.
    """

    c_scale = cm.rainbow(np.linspace(0, 1, len(categories)))
    c_dict = {}

    for i, c in zip(categories, c_scale):
        c_dict[i] = c

    return c_dict

# Widgets for settings
widg_sim       = widgets.IntText(value = 10, description = "Simulations:")
widg_algo      = widgets.Dropdown(options=["monte-carlo","q-learning"], value="monte-carlo", description="Algorithm:")
widg_new_model = widgets.Dropdown(options=[True,False], value=True, description="New Model:")
widg_comment   = widgets.Dropdown(options=[True,False], value=True, description="Show Game:")

# Widgets for parameters
widg_epsilon = widgets.FloatSlider(value=0.2, min=0, max=1, step=0.01, description="Epsilson:", readout_format=".2f")
widg_step    = widgets.FloatSlider(value=0.2, min=0, max=1, step=0.01, description="Step Size:", readout_format=".2f")

winners, turns, coverage = list(), list(), list()

# Agent parameters
agent_info = {"epsilon"  : widg_epsilon.value,
              "step_size": widg_step.value,
              "new_model": widg_new_model.value}


# Run simulations
run = ala.tournament(iterations = widg_sim.value,
                     algo       = widg_algo.value,
                     comment    = widg_comment.value,
                     agent_info = agent_info)


winners.extend(run[0])
turns.extend(run[1])
coverage.extend(run[2])

for i in winners:
    print(i.name)