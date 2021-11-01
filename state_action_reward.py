# 1. Libraries
# -------------------------------------------------------------------------

import pandas as pd
import numpy as np
import itertools


# 2. Functions
# -------------------------------------------------------------------------

def states():
    """
    Help text
    """

    # Normal cards
    norm_cards = {"PIR": 2, "ZOL": 2, "TOK": 2, "MAK": 2}

    # Special cards
    norm_cards_play = {"PIR#": 1, "ZOL#": 1, "TOK#": 1, "MAK#": 1}

    # Combine dictionaries
    states_dict = {**norm_cards, **norm_cards_play}
    states = [["PIR", "ZOL", "TOK", "MAK"]]

    for val in states_dict.values():
        aux = range(0, val + 1)
        states.append(aux)

    # Conduct all combinations
    states = list(itertools.product(*states))
    states_all = list()


    for i in range(len(states)):
        if (states[i][1] >= states[i][5]) and (states[i][2] >= states[i][6]) and (states[i][3] >= states[i][7]) and (
                states[i][4] >= states[i][8]):
            states_all.append(states[i])

    return states_all


def actions():
    """
    Help text
    """

    actions_all = ["PIR", "ZOL", "TOK", "MAK"]
    return actions_all


def rewards(states, actions):
    """
    Help text
    """

    R = np.zeros((len(states), len(actions)))
    states_t = [min(sum(states[i][1:5]), 1) for i in range(len(states))]

    for i in range(len(states)):
        if states_t[i] == 0:
            R[i] = 1

    R = pd.DataFrame(data=R,
                     columns=actions,
                     index=states)

    return R