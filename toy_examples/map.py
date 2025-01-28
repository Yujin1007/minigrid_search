import numpy as np

L = -3   #LAVA
S = -2    # switch
O = 0     #open space
W = -1    # wall
G = 3     #goal
R = 1     #red object
B = 2     #blue object
A = 4     #agent

DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3
STAY = 4


map = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 2,
    },
    "starting_pos": np.array([4, 1]),
    # "goal_pos": np.array([4, 6]),
    # "red_obj_pos": np.array([1, 6]),
    # "blue_obj_pos": np.array([7, 6]),
    "map_array": np.array([
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, W, O, O, B, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O],
            [W, O, W, W, W, W, G, W, W],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, W, O, O, B, O, O],
            [O, O, O, W, O, O, O, O, O]
        ]),
    # "ref_seq": [
    #     np.array([[1, 0, 0, 0]]),
    #     np.array([[0, 0, 0, 1]]),
    # ],
    # "obs_seqs": {
    #     0: {
    #         "descriptions": "Correct",
    #         "seq": [
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[0, 1, 0, 0]]),
    #             np.array([[0, 0, 1, 0]]),
    #             np.array([[0, 0, 0, 1]]),
    #         ]
    #     },
    #     1: {
    #         "descriptions": "Stuck at ref seq frame 1",
    #         "seq": [
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[0, 1, 0, 0]]),
    #         ]
    #     }
    # }
}
map2 = {
    "goal_pos": np.array([4, 6]),
    "map_array": np.array([
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, W, O, O, B, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O],
            [W, O, W, W, O, B, G, O, O],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, W, O, O, B, O, O],
            [O, O, O, W, O, O, O, O, O]
        ]),
}
empty_map = {
    "map_array": np.array([
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O],
            [W, O, W, W, W, W, G, W, W],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O]
    ]),
}
empty_map2 = {
    "map_array": np.array([
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [W, O, O, O, O, O, G, W, W],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O]
    ]),
}
empty_map3 = {
    "map_array": np.array([
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, O, O, O, B, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, A, O, O, O, O, G, B, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, O, O, B, O, O, O],
            [O, O, O, O, O, O, O, O, O]
    ]),
}

student_failed_map = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 2,
    },
    "starting_pos": np.array([3, 6]),
    "goal_pos": np.array([4, 6]),
    # "red_obj_pos": np.array([1, 6]),
    # "blue_obj_pos": np.array([7, 6]),
    "map_array": np.array([
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, W, O, O, A, O, O],
            [W, O, W, W, W, W, B, W, W],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, W, O, O, R, O, O],
            [O, O, O, W, O, O, O, O, O]
        ]),
    # "ref_seq": [
    #     np.array([[1, 0, 0, 0]]),
    #     np.array([[0, 0, 0, 1]]),
    # ],
    # "obs_seqs": {
    #     0: {
    #         "descriptions": "Correct",
    #         "seq": [
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[0, 1, 0, 0]]),
    #             np.array([[0, 0, 1, 0]]),
    #             np.array([[0, 0, 0, 1]]),
    #         ]
    #     },
    #     1: {
    #         "descriptions": "Stuck at ref seq frame 1",
    #         "seq": [
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[0, 1, 0, 0]]),
    #         ]
    #     }
    # }
}

map_realizability1 = {
    # "red_obj_pos": np.array([1, 6]),
    # "blue_obj_pos": np.array([7, 6]),
    "map_array": np.array([
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, W, O, O, O, B, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, O, O, B, O, O, O],
            [O, A, O, O, O, G, B, O, O],
            [O, O, O, W, O, B, O, O, O],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, W, O, O, O, O, O]
        ]),

}

map_realizability2 = {
    # "red_obj_pos": np.array([1, 6]),
    # "blue_obj_pos": np.array([7, 6]),
    "map_array": np.array([
            [O, O, O, W, O, O, O, O, O],
            [O, O, O, O, O, O, O, B, O],
            [O, O, O, A, O, O, O, O, O],
            [O, O, B, O, O, B, O, O, O],
            [O, O, B, O, O, G, O, O, O],
            [O, O, O, O, B, O, O, B, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, O, O, O, B, O, O],
            [O, O, O, W, O, O, O, O, O]
        ]),
}

map_realizability3 = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 2,
    },
    # "red_obj_pos": np.array([1, 6]),
    # "blue_obj_pos": np.array([7, 6]),
    "map_array": np.array([
            [O, O, O, O, O],
            [O, O, B, O, O],
            [O, B, G, O, O],
            [O, O, O, O, O],
            [O, O, O, O, O],
        ]),
    # "ref_seq": [
    #     np.array([[1, 0, 0, 0]]),
    #     np.array([[0, 0, 0, 1]]),
    # ],
    # "obs_seqs": {
    #     0: {
    #         "descriptions": "Correct",
    #         "seq": [
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[0, 1, 0, 0]]),
    #             np.array([[0, 0, 1, 0]]),
    #             np.array([[0, 0, 0, 1]]),
    #         ]
    #     },
    #     1: {
    #         "descriptions": "Stuck at ref seq frame 1",
    #         "seq": [
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[0, 1, 0, 0]]),
    #         ]
    #     }
    # }
}
student_failed_map3_1 = {
    "plot": {
        "reward_vmin": 0,
        "reward_vmax": 2,
    },
    "goal_pos": np.array([2,2]),
    # "red_obj_pos": np.array([1, 6]),
    # "blue_obj_pos": np.array([7, 6]),
    "map_array": np.array([
            [O, O, O, O, O],
            [O, O, R, O, O],
            [O, O, B, O, O],
            [O, O, O, O, O],
            [O, O, O, O, O],
        ]),
    # "ref_seq": [
    #     np.array([[1, 0, 0, 0]]),
    #     np.array([[0, 0, 0, 1]]),
    # ],
    # "obs_seqs": {
    #     0: {
    #         "descriptions": "Correct",
    #         "seq": [
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[0, 1, 0, 0]]),
    #             np.array([[0, 0, 1, 0]]),
    #             np.array([[0, 0, 0, 1]]),
    #         ]
    #     },
    #     1: {
    #         "descriptions": "Stuck at ref seq frame 1",
    #         "seq": [
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[1, 0, 0, 0]]),
    #             np.array([[0, 1, 0, 0]]),
    #         ]
    #     }
    # }
}

lava_switch = {
    "starting_pos" : np.array([2,0]),
    "map_array": [np.array([
                    [O, O, S, O, O, O, O, O, O],
                    [O, O, O, O, O, O, O, O, O],
                    [O, O, O, O, O, L, L, L, L],
                    [O, L, L, L, O, L, L, L, O],
                    [O, L, L, L, O, O, L, L, O],
                    [O, L, L, L, L, O, O, O, O],
                    [O, L, L, L, L, L, O, O, O],
                    [O, L, L, L, L, L, O, O, O],
                    [O, L, L, L, L, L, L, O, O],
                    [O, L, L, L, L, L, L, L, G],
                    [O, L, L, L, L, L, L, L, O],
                    [O, L, L, L, L, L, L, L, O],
                    [O, O, O, O, O, O, O, O, O],
            ]),np.array([
                    [O, O, S, O, O, O, O, O, O],
                    [O, O, O, O, O, O, O, O, O],
                    [O, O, O, O, L, L, L, L, L],
                    [O, L, L, O, O, L, L, L, O],
                    [O, L, L, L, O, O, L, L, O],
                    [O, L, L, L, L, O, O, O, O],
                    [O, L, L, L, L, L, O, O, O],
                    [O, L, L, L, L, L, O, O, O],
                    [O, L, L, L, L, L, L, O, O],
                    [O, L, L, L, L, L, L, L, G],
                    [O, L, L, L, L, L, L, L, O],
                    [O, L, L, L, L, L, L, L, O],
                    [O, O, O, O, O, O, O, O, O],
            ]),np.array([
                    [O, O, S, O, O, O, O, O, O],
                    [O, O, O, O, O, O, O, O, O],
                    [O, O, O, O, L, L, L, L, L],
                    [O, L, O, L, O, O, O, O, O],
                    [O, L, O, L, O, O, O, O, O],
                    [O, L, O, O, O, O, O, O, O],
                    [O, L, L, L, L, L, O, O, O],
                    [O, L, L, L, L, L, O, O, O],
                    [O, L, L, L, L, L, L, O, O],
                    [O, L, L, L, L, L, L, L, G],
                    [O, L, L, L, L, L, L, L, O],
                    [O, L, L, L, L, L, L, L, O],
                    [O, O, O, O, O, O, O, O, O],
            ]),

            ]
}
lava_switch1 = {
    "map_array": np.array([
            [O, O, S, O, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, O, O, L, L, L, O],
            [O, L, L, L, O, L, L, L, O],
            [O, L, L, L, O, O, L, L, O],
            [O, L, L, L, L, O, O, O, O],
            [O, L, L, L, L, L, O, O, O],
            [O, L, L, L, L, L, O, O, O],
            [O, L, L, L, L, L, L, O, O],
            [O, L, L, L, L, L, L, L, G],
            [O, L, L, L, L, L, L, L, O],
            [O, L, L, L, L, L, L, L, O],
            [O, O, O, O, O, O, O, O, O],
    ]),
}
lava_switch2 = {
    "map_array": np.array([
            [O, O, S, O, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, O, L, L, L, L, O],
            [O, L, L, O, O, L, L, L, O],
            [O, L, L, L, O, O, L, L, O],
            [O, L, L, L, L, O, O, O, O],
            [O, L, L, L, L, L, O, O, O],
            [O, L, L, L, L, L, O, O, O],
            [O, L, L, L, L, L, L, O, O],
            [O, L, L, L, L, L, L, L, G],
            [O, L, L, L, L, L, L, L, O],
            [O, L, L, L, L, L, L, L, O],
            [O, O, O, O, O, O, O, O, O],
    ]),
}
lava_switch3 = {
    "map_array": np.array([
            [O, O, S, O, O, O, O, O, O],
            [O, O, O, O, O, O, O, O, O],
            [O, O, O, O, L, L, L, L, O],
            [O, L, O, L, O, O, O, O, O],
            [O, L, O, L, O, O, O, O, O],
            [O, L, O, O, O, O, O, O, O],
            [O, L, L, L, L, L, O, O, O],
            [O, L, L, L, L, L, O, O, O],
            [O, L, L, L, L, L, L, O, O],
            [O, L, L, L, L, L, L, L, G],
            [O, L, L, L, L, L, L, L, O],
            [O, L, L, L, L, L, L, L, O],
            [O, O, O, O, O, O, O, O, O],
    ]),
}