from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Iterable
import itertools

# Corner position indices:
# 0: URF, 1: UFL, 2: ULB, 3: UBR, 4: DFR, 5: DLF, 6: DBL, 7: DRB
NUM_CORNERS = 8
#edge position indices:
# 0: UR, 1: UF, 2: UL, 3: UB, 4: DR, 5: DF, 6: DL, 7: DB, 8: FR, 9: FL, 10: BL, 11: BR
NUM_EDGES = 12


MOVE_CORNERS = {
    "U":  ((3, 0, 1, 2, 4, 5, 6, 7), (0, 0, 0, 0, 0, 0, 0, 0)),
    "U2": ((2, 3, 0, 1, 4, 5, 6, 7), (0, 0, 0, 0, 0, 0, 0, 0)),
    "U'": ((1, 2, 3, 0, 4, 5, 6, 7), (0, 0, 0, 0, 0, 0, 0, 0)),

    "D":  ((0, 1, 2, 3, 5, 6, 7, 4), (0, 0, 0, 0, 0, 0, 0, 0)),
    "D2": ((0, 1, 2, 3, 6, 7, 4, 5), (0, 0, 0, 0, 0, 0, 0, 0)),
    "D'": ((0, 1, 2, 3, 7, 4, 5, 6), (0, 0, 0, 0, 0, 0, 0, 0)),

    "R":  ((4, 1, 2, 0, 7, 5, 6, 3), (2, 0, 0, 1, 1, 0, 0, 2)),
    "R2": ((7, 1, 2, 4, 3, 5, 6, 0), (0, 0, 0, 0, 0, 0, 0, 0)),
    "R'": ((3, 1, 2, 7, 0, 5, 6, 4), (1, 0, 0, 2, 2, 0, 0, 1)),

    "L":  ((0, 2, 6, 3, 4, 1, 5, 7), (0, 1, 2, 0, 0, 2, 1, 0)),
    "L2": ((0, 6, 5, 3, 4, 2, 1, 7), (0, 0, 0, 0, 0, 0, 0, 0)),
    "L'": ((0, 5, 1, 3, 4, 6, 2, 7), (0, 2, 1, 0, 0, 1, 2, 0)),

    "F":  ((1, 5, 2, 3, 0, 4, 6, 7), (1, 2, 0, 0, 2, 1, 0, 0)),
    "F2": ((5, 4, 2, 3, 1, 0, 6, 7), (0, 0, 0, 0, 0, 0, 0, 0)),
    "F'": ((4, 0, 2, 3, 5, 1, 6, 7), (2, 1, 0, 0, 1, 2, 0, 0)),

    "B":  ((0, 1, 3, 7, 4, 5, 2, 6), (0, 0, 1, 2, 0, 0, 2, 1)),
    "B2": ((0, 1, 7, 6, 4, 5, 3, 2), (0, 0, 0, 0, 0, 0, 0, 0)),
    "B'": ((0, 1, 6, 2, 4, 5, 7, 3), (0, 0, 2, 1, 0, 0, 1, 2)),
}
MYMOVE = MOVE_CORNERS["R"]
        
MOVE_EDGES = {
    "U":  ((3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11), (0,)*12),
    "U2": ((2, 3, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11), (0,)*12),
    "U'": ((1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11), (0,)*12),

    "D":  ((0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11), (0,)*12),
    "D2": ((0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11), (0,)*12),
    "D'": ((0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11), (0,)*12),

    "R":  ((0, 1, 2, 3, 11, 5, 6, 8, 4, 9, 10, 7), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    "R2": ((0, 1, 2, 3, 8, 5, 6, 4, 11, 9, 10, 7), (0,)*12),
    "R'": ((0, 1, 2, 3, 8, 5, 6, 11, 7, 9, 10, 4), (0,)*12),

    "L":  ((0, 1, 10, 3, 4, 5, 2, 7, 8, 6, 9, 11), (0,)*12),
    "L2": ((0, 1, 9, 3, 4, 5, 10, 7, 8, 2, 6, 11), (0,)*12),
    "L'": ((0, 1, 6, 3, 4, 5, 9, 7, 8, 10, 2, 11), (0,)*12),

    "F":  ((0, 5, 2, 3, 1, 9, 6, 7, 8, 5, 10, 11),  # orientation flips on F\,B
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1)), #need to change this.
    "F2": ((0, 8, 2, 5, 4, 1, 6, 9, 5, 4, 10, 11), (0,)*12),
    "F'": ((0, 4, 2, 3, 9, 1, 6, 7, 8, 5, 10, 11),
            (0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0)),

    "B":  ((7, 1, 2, 6, 4, 5, 10, 11, 8, 9, 3, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1)),
    "B2": ((11, 1, 2, 10, 4, 5, 3, 6, 8, 9, 7, 0), (0,)*12),
    "B'": ((3, 1, 2, 0, 4, 5, 6, 10, 8, 9, 11, 7),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1)),
}
  
def inverse_corner(move):
    ''' Inverts the corner permutation represented by the tuple. 
    "U":  ((3, 0, 1, 2, 4, 5, 6, 7), (0, 0, 0, 0, 0, 0, 0, 0)),
    
    '''
    perm = move[0]
    ori = move[1]
    new_perm = ( 
        perm.index(0), perm.index(1), perm.index(2), perm.index(3), perm.index(4), perm.index(5), perm.index(6), perm.index(7)
    )
    my_list = []
    for element in ori:
        if element == 0:
            my_list.append(0)
        elif element == 1:
            my_list.append(2)
        elif element == 2:
            my_list.append(1)
    new_ori = tuple(my_list)
    return (new_perm, new_ori)



def double_up_corner(move):
    '''
    prodouces the double of the input move.

    "U":  ((3, 0, 1, 2, 4, 5, 6, 7)
    '''
    my_tuple = (
        move.index(move.index(0)),  move.index(move.index(1)), move.index(move.index(2)), 
        move.index(move.index(3)), move.index(move.index(4)), move.index(move.index(5)), 
        move.index(move.index(6)),move.index(move.index(7))
    )
    return my_tuple



if __name__ == "__main__":
    print(inverse_corner(MYMOVE))