import random
import subprocess

corner_cases = {
    "0c0": "",
    "0c3": "R B2 R2 D2 R' B2 R2 B2 D2 R",
    "0c4": "U' R2 U R2 U2 B2 U B2 U",
    "2c3": "U F2 U2 R2 F2 U' F2 R2 U",
    "2c4": "U2 R2 U L2 U R2 U' L2 U",
    "2c5": "U L2 U R2 U' R2 U R2 U",
    "4a1": "U R2 L2 U F2 B2 D",
    "4a2": "U R2 U2 F2 U' F2 U F2 U2 R2 U",
    "4a3": "U' F2 R2 U L2 U2 B2 D",
    "4a4": "U R2 U' F2 U R2 F2 U B2 U' F2 U",
    "4b2": "U R2 F2 R2 F2 U",
    "4b3": "U R2 U2 B2 U' F2 L2 D",
    "4b4": "U R2 U' R2 F2 U2 F2 U' F2 U",
    "4b5": "U B2 U B2 U' F2 U F2 U",
}

edge_cases = {
    "0e": "",
    "2e": "U R2 F2 R2 U",
    "4e": "U L2 R2 D",
    "6e": "U L2 R2 B2 L2 B2 D",
    "8e": "U R2 L2 F2 B2 U",
}
def get_dr_in_subset(corners, edges):
    '''Genereates a random element of the dr subset 
    specified by the corners and edges arguments'''

    pre_half_turns = random.choices(['R2', 'L2', 'B2', 'F2', 'U2', 'D2'], k = 25)
    post_half_turns = random.choices(['R2', 'L2', 'B2', 'F2', 'U2', 'D2'], k = 25)
    dr = (" ").join(pre_half_turns) + " "+ corner_cases[corners] + " "+ edge_cases[edges] + " " + (" ").join(post_half_turns)
    inverse = subprocess.check_output(["nissy", "solve", "-p", dr])
    return (subprocess.check_output(["nissy", "invert", inverse])).decode("utf-8").strip()

def get_subset_to_optimal_dict():
    subset_dict = {}
    for cc in corner_cases:
        for ec in edge_cases:
            for i in range(5):
                dr = get_dr_in_subset(cc, ec)
                optimal_length = dr.count(" ") + 1 #number of spaces + 1
                subset_dict.setdefault(cc + " " +ec, []).append(optimal_length)
            print(subset_dict)
    return subset_dict

def get_corner_optimal_to_optimal_dict():
    corner_opt_dict = {}
    dr = get_dr_in_subset


            
#def get_random_dr():
if __name__ == "__main__":
    print(get_subset_to_optimal_dict())