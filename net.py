
import numpy as np
from dr_to_solved import state
from dr_to_solved import helpers
#stickers
white = set([4, 5, 6, 9, 10, 13, 14, 15])
yellow = set([24, 25, 26, 29, 30, 33, 34, 35])
red = set([7, 11, 16, 27, 31, 36])
green = set([17, 18, 19, 37, 38, 39])
orange = set([3, 8, 12, 23, 28, 32])
blue = set([0, 1, 2, 20, 21, 22])




corner_id_to_stickers= {
    0: [15, 16, 19],
    1: [13, 17, 12],
    2: [4, 3, 0],
    3: [6, 2, 7], 
    4: [35, 39, 36],
    5: [33, 32, 37],
    6: [24, 20, 23],
    7: [26, 27, 22]
}
edge_id_to_stickers = {
    0: [10, 11], 
    1: [14, 18], 
    2: [9, 8], 
    3: [5, 1], 
    8: [30, 31], 
    9: [34, 38], 
    10: [29, 28], 
    11: [25, 21], 
}

COLOUR_TO_INDEX = {
    'white': 0,
    'green': 1,
    'red': 2,
    'blue': 3,
    'orange': 4,
    'yellow': 5
}
INDEX_TO_COLOUR = {v: k for k, v in COLOUR_TO_INDEX.items()}


CUBE_TO_STICKERS = {0: [15, 16, 19], 1: [13, 17, 12], 2: [4, 3, 0], 3: [6, 2, 7], 
                    4: [35, 39, 36], 5: [33, 32, 37], 6: [24, 20, 23], 7: [26, 27, 22], 
                    8: [10, 11], 9: [14, 18], 10: [9, 8], 11: [5, 1], 16: [30, 31], 
                    17: [34, 38], 18: [29, 28], 19: [25, 21]}

PAIR_TO_CUBES = {0: [0, 8], 1: [1, 9], 2: [2, 10], 3: [3, 11], 4: [4, 17], 
                 5: [5, 18], 6: [6, 19], 7: [7, 16], 8: [0, 9], 9: [1, 10], 
                 10: [2, 11], 11: [3, 8], 12: [4, 16], 13: [5, 17], 
                 14: [6, 18], 15: [7, 19], 16: [0, 4], 17: [1, 5], 
                 18: [2, 6], 19: [3, 7],20: [8, 16], 21: [9, 17], 
                 22: [10, 18], 23: [11, 19]} #a -> b-> vertical corner pairs -> vertical edge pairs.
TRIPLE_TO_PAIRS = { #top Ls -> bottom Ls -> lines -> hooks_involving_pair_16 -> hooks_17 -> hooks_18 -> hooks_19
    0: [0, 8], #0-3 top L (0th a pair and 0th b pair) c0
    1: [1, 9], #surrounding c1
    2: [2, 10], #c2
    3: [3, 11], #c3
    4: [4, 12], #4-7 bot L              c4
    5: [5, 13], #c5
    6: [6, 14], #c6
    7: [7, 15], #c7
    8: [8, 1], #8-11 top line
    9: [9,2],
    10: [10,3],
    11: [11,0],
    12: [13,4], #12-15 bot line
    13: [14,5],
    14: [15,6],
    15: [12,7],
    16: [16,0], #16-19 involving pair 16 (FR)
    17: [16,8],
    18: [16,4],
    19: [16,12],
    20: [17,1], #20-23 involving pair 17 (FL)
    21: [17,9],
    22: [17,5],
    23: [17,13],
    24: [18,2], #24-27 involving pair 18 (BL)
    25: [18,10],
    26: [18,6],
    27: [18,14],
    28: [19,3], #28-31 involving pair 19 (BR)
    29: [19,11],
    30: [19,7], 
    31: [19,15],
    32: [20, 0], #32-35 involving pair 20 (RR)
    33: [20, 11],
    34: [20, 7],
    35: [20, 12], 
    36: [21, 1], #36-39 involving pair 21 (FF)
    37: [21, 8],
    38: [21, 4],
    39: [21, 13],
    40: [22, 2], #40-43 involving pair 22 (LL)
    41: [22, 9],
    42: [22, 5],
    43: [22, 14],
    44: [23, 3], #44-47 involving pair 23 (BB)
    45: [23, 10],
    46: [23, 6],
    47: [23, 15]

}
QUADRUPLE_TO_TRIPLES = {
    0: [17, 18, 37, 38], #front, right     #0-8 the side squares, ordered counterclockwise, starting from front,right
    1: [20, 23, 36, 39], #front,left
    2: [21, 22, 41, 42], #left, front
    3: [24, 27, 40, 43], #left, back
    4: [25, 26, 45, 46], #back, left
    5: [28, 31, 44, 47], #back, right
    6: [29, 30, 33, 34], #right, back
    7: [16, 19, 32, 35], #right, front
    8: [0, 8], #"Boot a0 (ja perm, setup: L' U' L2 F L' U' L' U L F' L' U L U') (pairs 0 and 1)"
    9: [1, 9], # pairs 1 and 2 // L surrounding c1 and line 1
    10: [2, 10], # pairs 2 and 3
    11: [3, 11], # pairs 3 and 0
    12: [4, 12], # pairs 4 and 5
    13: [5, 13], # pairs 5 and 6
    14: [6, 14], # pairs 6 and 7
    15: [7, 15], # pairs 7 and 4
    16: [1, 8], #Boot b0 (pairs 8 and 9) 
    17: [2, 9],
    18: [3, 10],
    19: [0, 11], #standard jb perm
    20: [4, 12], #c4 and bottom front line
    21: [5, 13],
    22: [6, 14],
    23: [7, 15],
    24: [0, 16, 17], #extended c0 in all directions, involves 3 triples (one L and two hooks)
    25: [1, 20, 21], #c1
    26: [2, 24, 25], #c2
    27: [3, 28, 29], #c3
    28: [4, 18, 19], #c4
    29: [5, 22, 23], #c5
    30: [6, 26, 27], #c6
    31: [7, 30, 31], #c7
    32: [16, 18], # two zig-zags around each corner column. Ordered by column 16-19, then S coming before Z shape.
    33: [17, 19],
    34: [20, 22],
    35: [21, 23],
    36: [24, 26],
    37: [25, 27],
    38: [28, 30],
    39: [29, 31]
}

def printy():
    for i in range(32, 48):
        print(str(i) + ": [],")
def print_cube_to_stickers():
    mydict = {}
    for corner in corner_id_to_stickers:
        mydict[corner] = corner_id_to_stickers[corner]
    for edge in edge_id_to_stickers:
        mydict[edge + 8] = edge_id_to_stickers[edge]
    print(mydict)
def print_pair_to_cubes():
    mydict = {}
    for index, (corner, edge) in enumerate(state.PAIRS["a"]):
        mydict[index] = [corner, edge +8]
    for index, (corner, edge) in enumerate(state.PAIRS["b"]):
        mydict[index+8] = [corner, edge +8]
    mydict[16] = [0, 4] #start with front-right vertical corner pair
    mydict[17] = [1, 5]
    mydict[18] = [2, 6]
    mydict[19] = [3, 7]
    mydict[20] = [8, 16] #start with right-middle vertical edge pair
    mydict[21] = [9, 17]
    mydict[22] = [10, 18]
    mydict[23] = [11, 19]
    print(mydict)



def sticker_colour(sticker):
    if sticker in white:
        return COLOUR_TO_INDEX['white']
    if sticker in yellow:
        return COLOUR_TO_INDEX['yellow']
    if sticker in red:
        return COLOUR_TO_INDEX['red']
    if sticker in green:
        return COLOUR_TO_INDEX['green']
    if sticker in orange:
        return COLOUR_TO_INDEX['orange']
    if sticker in blue:
        return COLOUR_TO_INDEX['blue']
    raise ValueError(f"Sticker {sticker} not recognized.")

def scramble_to_onehot(scramble):
    cube = state.State()
    cube.apply(scramble)

    c_perm = cube.corners[0]
    e_perm = cube.edges[0]
    onehot = np.zeros((240,), dtype=np.float32)


    for position, present in enumerate(c_perm): 
        sticker_positions = corner_id_to_stickers[position]
        stickers_present = corner_id_to_stickers[present]

        for pos_sticker, actual_sticker in zip(sticker_positions, stickers_present): 
            colour_idx = sticker_colour(actual_sticker)
            index = (pos_sticker * 6) + colour_idx # 
            onehot[index] = 1.0


    for position, present in enumerate(e_perm):
        if not position in edge_id_to_stickers: #skip edges in the middle slice
            continue
        sticker_positions = edge_id_to_stickers[position]
        stickers_present = edge_id_to_stickers[present]

        for pos_sticker, actual_sticker in zip(sticker_positions, stickers_present): 
            colour_idx = sticker_colour(actual_sticker)
            index = (pos_sticker * 6) + colour_idx 
            onehot[index] = 1.0

    return onehot
    #the assumption is that every 6 indices would have exactly one 1, because every sticker is present once.

def check_blocks(arr):
    #sanity check 
    assert len(arr) % 6 == 0, "Array length must be divisible by 6."
    blocks = arr.reshape(-1, 6)
    return np.all(blocks.sum(axis=1) == 1)

def decode_onehot(onehot):
    #return list of ordered stickers present in canonical ordering
    stickers = []
    for i in range(0, len(onehot), 6):
        block = onehot[i:i+6]
        if np.sum(block) != 1:
            stickers.append("INVALID")
        else:
            colour_index = np.argmax(block)
            stickers.append(INDEX_TO_COLOUR[int(colour_index)])
    return stickers


'''
canonical ordering of all 20 pieces:
0->7: corners based on their ids. 
8->11: edges 0 to 3
12->15: edges 8 to 11

'''

'''
canonical ordering of all 20 pairs: 
0 -> 7: "a" pairs in the order I defined them in state.py
8 -> 15: "b"...
16: c0-c4
17: c1-c5
18: c2-c6
19: c3-c7
'''
'''
canonical ordering of all triples (lines, Ls, hooks):

'''

'''
canonical ordering of all quadruples (extended corner, long L, side square):



'''
printy()
print_pair_to_cubes()
'''
scramble = helpers.get_dr_scrambles(1)[0]
print(scramble)
onehot = scramble_to_onehot(scramble)
print(onehot)
print(check_blocks(onehot))
'''




