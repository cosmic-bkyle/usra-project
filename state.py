from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Iterable
import itertools
import networkx as nx
import matplotlib.pyplot as plt


# Corner position indices:
# 0: URF, 1: UFL, 2: ULB, 3: UBR, 4: DFR, 5: DLF, 6: DBL, 7: DRB
NUM_CORNERS = 8
#edge position indices:
# 0: UR, 1: UF, 2: UL, 3: UB, 4: DR, 5: DF, 6: DL, 7: DB, 8: FR, 9: FL, 10: BL, 11: BR
NUM_EDGES = 12

ID_CORNERS = ((0, 1, 2, 3, 4, 5, 6, 7), (0, 0, 0, 0, 0, 0, 0, 0))
ID_EDGES = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), (0,)*12)

CORNER_ADJ_EDGES = {
    0: (0, 1, 4),      
    1: (1, 2, 5),      
    2: (2, 3, 6),     
    3: (3, 0, 7),     
    4: (4, 8, 9),  
    5: (5, 9, 10), 
    6: (6, 10, 11),    
    7: (7, 8, 11) 
}
CORNER_ADJ_EDGES_NO_SLICE = {
    0: (0, 1),
    1: (1, 2),
    2: (2, 3),
    3: (3, 0),
    4: (8, 9),
    5: (9, 10),
    6: (10, 11),
    7: (11, 8)
}



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

    "D":  ((0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 8), (0,)*12),
    "D2": ((0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 8, 9), (0,)*12),
    "D'": ((0, 1, 2, 3, 4, 5, 6, 7, 11, 8, 9, 10), (0,)*12),

    "R":  ((4, 1, 2, 3, 8, 5, 6, 0, 7, 9, 10, 11), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    "R2": ((8, 1, 2, 3, 7, 5, 6, 4, 0, 9, 10, 11), (0,)*12),
    "R'": ((7, 1, 2, 3, 0, 5, 6, 8, 4, 9, 10, 11), (0,)*12),

    "L":  ((0, 1, 6, 3, 4, 2, 10, 7, 8, 9, 5, 11), (0,)*12),
    "L2": ((0, 1, 10, 3, 4, 6, 5, 7, 8, 9, 2, 11), (0,)*12),
    "L'": ((0, 1, 5, 3, 4, 10, 2, 7, 8, 9, 6, 11), (0,)*12),

    "F":  ((0, 5, 2, 3, 1, 9, 6, 7, 8, 4, 10, 11),  # orientation flips on F\,B
            (0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0)), #need to change this.
    "F2": ((0, 9, 2, 3, 5, 4, 6, 7, 8, 1, 10, 11, 11), (0,)*12),
    "F'": ((0, 4, 2, 3, 9, 1, 6, 7, 8, 5, 10, 11),
            (0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0)),

    "B":  ((0, 1, 2, 7, 4, 5, 3, 11, 8, 9, 10, 6),
            (0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1)),
    "B2": ((0, 1, 2, 11, 4, 5, 7, 6, 8, 9, 10, 3), (0,)*12),
    "B'": ((0, 1, 2, 6, 4, 5, 11, 3, 8, 9, 10, 7), 
           (0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1)),
}
INDEX_TO_COLOURS = {
    # Corners (IDs 0–7)
    0: "WRG",
    1: "WGO",
    2: "WOB",
    3: "WBR",
    4: "YGR",
    5: "YOG",
    6: "YBO",
    7: "YRB",

    # Edges (IDs 8–19)
    8:  "WR",
    9:  "WG",
    10: "WO",
    11: "WB",
    12: "GR",
    13: "GO",
    14: "BO",
    15: "BR",
    16: "YR",
    17: "YG",
    18: "YO",
    19: "YB"
}
PAIRS = { #(cornerID, edgeID that touch. We have two "styles" of pairs: a and b. )
    "a": [(0, 0), (1, 1), (2, 2), (3, 3), (4, 9), (5, 10), (6, 11), (7, 8)],
    "b": [(0, 1), (1, 2), (2, 3), (3, 0), (4, 8), (5, 9), (6, 10), (7, 11)]
}
Ls = { # corner, edge 1, edge2
    "top": [(0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 0)],
    "bottom": [(4, 9, 8), (5, 10, 9), (6, 11, 10), (7, 8, 11)]
}
LINES = [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 0, 0), (5, 4, 9), (6, 5, 10), (7, 6, 11), (4, 7, 8)]# format (corner1, corner2, edge)

  
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
def inverse_edge(move):
    '''
    "F":  ((0, 5, 2, 3, 1, 9, 6, 7, 8, 5, 10, 11),  # orientation flips on F\,B
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1))
            
            '''
    perm = move[0]
    new_perm = ( 
        perm.index(0), perm.index(1), perm.index(2), perm.index(3), 
        perm.index(4), perm.index(5), perm.index(6), perm.index(7),
        perm.index(8), perm.index(9), perm.index(10), perm.index(11)
    )
    return (new_perm, move[1])

def inverse_move(move_symbol):
    ''' Returns the tuple corresponding to the inverse move symbol of that provided. '''
    return (inverse_corner(MOVE_CORNERS[move_symbol]), inverse_edge(MOVE_EDGES[move_symbol]))





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

    

class Cube:
    def __init__(self, moves = ""):
        self.corners = ID_CORNERS
        self.edges = ID_EDGES
    
    @staticmethod  
    def new_corners(prev_corners, move):
        prev_perm = prev_corners[0]
        new_perm = (prev_perm[MOVE_CORNERS[move][0][0]], prev_perm[MOVE_CORNERS[move][0][1]],
                    prev_perm[MOVE_CORNERS[move][0][2]],prev_perm[MOVE_CORNERS[move][0][3]],
                    prev_perm[MOVE_CORNERS[move][0][4]],prev_perm[MOVE_CORNERS[move][0][5]],
                    prev_perm[MOVE_CORNERS[move][0][6]],prev_perm[MOVE_CORNERS[move][0][7]])
        #FOR each index, you have ori + ori mod 3
        prev_ori = prev_corners[1]
        new_ori = ((MOVE_CORNERS[move][1][0] + prev_ori[MOVE_CORNERS[move][0][0]]) % 3, 
                   ((MOVE_CORNERS[move][1][1] + prev_ori[MOVE_CORNERS[move][0][1]]) % 3),
                   ((MOVE_CORNERS[move][1][2] + prev_ori[MOVE_CORNERS[move][0][2]]) % 3),
                   ((MOVE_CORNERS[move][1][3] + prev_ori[MOVE_CORNERS[move][0][3]]) % 3),
                   ((MOVE_CORNERS[move][1][4] + prev_ori[MOVE_CORNERS[move][0][4]]) % 3),
                   ((MOVE_CORNERS[move][1][5] + prev_ori[MOVE_CORNERS[move][0][5]]) % 3),
                   ((MOVE_CORNERS[move][1][6] + prev_ori[MOVE_CORNERS[move][0][6]]) % 3),
                   ((MOVE_CORNERS[move][1][7] + prev_ori[MOVE_CORNERS[move][0][7]]) % 3)
                
        )
        return (new_perm, new_ori)

    
    @staticmethod
    def new_edges(prev_edges, move):
        prev_perm = prev_edges[0]
        new_perm = (prev_perm[MOVE_EDGES[move][0][0]], prev_perm[MOVE_EDGES[move][0][1]],
                    prev_perm[MOVE_EDGES[move][0][2]],prev_perm[MOVE_EDGES[move][0][3]],
                    prev_perm[MOVE_EDGES[move][0][4]],prev_perm[MOVE_EDGES[move][0][5]],
                    prev_perm[MOVE_EDGES[move][0][6]],prev_perm[MOVE_EDGES[move][0][7]],
                    prev_perm[MOVE_EDGES[move][0][8]],prev_perm[MOVE_EDGES[move][0][9]],
                    prev_perm[MOVE_EDGES[move][0][10]],prev_perm[MOVE_EDGES[move][0][11]])
        #for orientation
        prev_ori = prev_edges[1]
        new_ori = (abs(MOVE_EDGES[move][1][0] - prev_ori[MOVE_EDGES[move][0][0]]), 
                   abs(MOVE_EDGES[move][1][1] - prev_ori[MOVE_EDGES[move][0][1]]), 
                   abs(MOVE_EDGES[move][1][2] - prev_ori[MOVE_EDGES[move][0][2]]), 
                   abs(MOVE_EDGES[move][1][3] - prev_ori[MOVE_EDGES[move][0][3]]), 
                   abs(MOVE_EDGES[move][1][4] - prev_ori[MOVE_EDGES[move][0][4]]), 
                   abs(MOVE_EDGES[move][1][5] - prev_ori[MOVE_EDGES[move][0][5]]), 
                   abs(MOVE_EDGES[move][1][6] - prev_ori[MOVE_EDGES[move][0][6]]), 
                   abs(MOVE_EDGES[move][1][7] - prev_ori[MOVE_EDGES[move][0][7]]), 
                   abs(MOVE_EDGES[move][1][8] - prev_ori[MOVE_EDGES[move][0][8]]), 
                   abs(MOVE_EDGES[move][1][9] - prev_ori[MOVE_EDGES[move][0][9]]), 
                   abs(MOVE_EDGES[move][1][10] - prev_ori[MOVE_EDGES[move][0][10]]), 
                   abs(MOVE_EDGES[move][1][11] - prev_ori[MOVE_EDGES[move][0][11]]))
        return (new_perm, new_ori)
            
        

    def apply(self, moves):
        movelist = moves.split(" ")
        for move in movelist:
            #self.corners = apply_corners(move)
            self.corners = Cube.new_corners(self.corners, move)
            self.edges = Cube.new_edges(self.edges, move)

        return (self.corners, self.edges)

    def adjacency_graph(self):
        """
        Return an undirected bipartite graph whose nodes are
          0‑7  = corner‑cubie IDs
          8‑19 = edge‑cubie IDs (edgeID + 8)
        An edge means “these two pieces are touching”.
        """
        G = nx.Graph()
        cperm, _ = self.corners    
        eperm, _ = self.edges    

        G.add_nodes_from(range(8),  part='corner') 
        G.add_nodes_from(range(8, 20), part='edge')

        for cpos, adj_edges in CORNER_ADJ_EDGES.items(): #for each location
            corner_cubie = cperm[cpos] #check which piece is in that location
            for epos in adj_edges:   #for each touching edge location  
                edge_cubie   = eperm[epos] #check what's in that spot
                G.add_edge(corner_cubie, edge_cubie +8) #add a connection in G.
        return G 
    

    @staticmethod
    def get_bips(scrambles):
        '''returns the list of adjacency graphs corresponding to the given scrambles'''
        bips = []
        for scramble in scrambles:
            newcube = Cube()
            newcube.apply(scramble)
            bips.append(newcube.adjacency_graph())
        return bips
    
    def amt_pairs(self):
        '''returns a tuple of the number of a-pairs and b-pairs present in the domino layers'''
        a_pairs = 0
        b_pairs = 0
        for a_spot in PAIRS["a"]:
            (corner_spot, edge_spot) = a_spot #unpack the tuple
            if (self.corners[0][corner_spot], self.edges[0][edge_spot]) in PAIRS["a"]:
                a_pairs += 1
        for b_spot in PAIRS["b"]:
            (corner_spot, edge_spot) = b_spot
            if (self.corners[0][corner_spot], self.edges[0][edge_spot]) in PAIRS["b"]:
                b_pairs += 1
        return a_pairs, b_pairs
    def amt_lines(self):
        ''' returns the number of corner-edge-corner lines.
        '''
        lines = 0
        for (c1, c2, e)  in LINES: #(corner, edge, edge)
            mytuple = (self.corners[0][c1], self.corners[0][c2], self.edges[0][e]) 
            if mytuple in LINES:
                lines += 1
        return lines

        
    def amt_Ls(self):
        '''returns tuples of # good Ls and # bad Ls. Ls are encoded c, e1, e2 where e1 e2 are ordered clockwise 
        '''
        good_Ls = 0
        bad_Ls = 0
        for (t1, t2, t3) in Ls["top"]:
            triple  = (self.corners[0][t1], self.edges[0][t2], self.edges[0][t3])
            if triple in Ls["top"]:
                good_Ls += 1
            elif triple in Ls["bottom"]:
                bad_Ls += 1
                
        for (t1, t2, t3) in Ls["bottom"]:
            triple  = (self.corners[0][t1], self.edges[0][t2], self.edges[0][t3])   
            if triple in Ls["top"]:
                bad_Ls += 1
            elif triple in Ls["bottom"]:
                good_Ls += 1
        return good_Ls, bad_Ls

        



    
    @staticmethod
    def draw_adjacency_graph(G):
        '''
        visualize the given adjacency graph.
        '''
        corner_nodes = [n for n, d in G.nodes(data=True) if d['part'] == 'corner']
        edge_nodes = [n for n, d in G.nodes(data=True) if d['part'] == 'edge']
        pos = nx.bipartite_layout(G, corner_nodes)

        node_drawing_colors = ['lightgreen' if n in corner_nodes else 'lightskyblue' for n in G.nodes()]
        labels = {node: INDEX_TO_COLOURS.get(node) for node in G.nodes()}

        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, labels = labels, node_color=node_drawing_colors, node_size=2000)
        plt.title("Cube Adjacency Graph (Bipartite Layout)")
        plt.show()


def main():
    '''
    visualize bipartite adjacency graph of input scramble
    '''
    mycube = Cube()
    mycube.apply("L2 U' D B2 L2 U' D F2 U' R2 U F2 U' R2 B2")
    

if __name__ == "__main__":
    main()


