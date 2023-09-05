import numpy as np


def reset():
    state_place_0, state_place_1 = None, None
    return state_place_0, state_place_1


def get_initial_state():
    state_0 = None
    return state_0


def step_place(state, action):
    next_state = None
    reward = None
    placed = None
    return next_state, reward, placed


def step_game(state, action):
    next_state = None
    reward = None
    final = None
    return next_state, reward, final
