''' a framework for scoring the "shortness" and "abundance" of EO reductions on all 3 axes x 2 sides = 6 "regions" of search.'''
from dr_to_solved import helpers
import numpy as np
import pandas as pd
from pathlib import Path
import csv
S = 50 #num of scrambles
N = 10 #how many eos to look at on each axis.
M = 3 #how many drs to check for on each EO.
AXES = ["FB", "RL", "UD"]
ADJ = {"FB": ["RL", "UD"], "RL": ["FB", "UD"], "UD": ["FB", "RL"]}

def main():
    scrambles = helpers.get_full_scrambles(S)

    #normal_drs = helpers.get_optimal_drs_normal(scrambles) #returns two lists: 
    print(scrambles)
    eosfb = helpers.get_n_eos_axis("fb", N, scrambles) # length S.
    drud_eofb = helpers.get_m_drs_axis("fb", "ud", N, M, scrambles, eosfb) # length SxN, needs to be looped through for each scramble.
    drrl_eofb = helpers.get_m_drs_axis("fb", "rl", N, M, scrambles, eosfb)
    

    eosrl = helpers.get_n_eos_axis("rl", N, scrambles)
    drud_eorl = helpers.get_m_drs_axis("rl", "ud", N, M, scrambles, eosrl)
    drfb_eorl =  helpers.get_m_drs_axis("rl", "fb", N, M, scrambles, eosrl)

    eosud = helpers.get_n_eos_axis("ud", N, scrambles)
    drfb_eoud = helpers.get_m_drs_axis("ud", "fb", N, M, scrambles, eosud)
    drrl_eoud = helpers.get_m_drs_axis("ud", "rl", N, M, scrambles, eosud)

    y = [{}]* S
    for i in range(S):
        results = []
        for A in AXES:
            eos = []
            if A == "FB":
                eos = eosfb
            elif A == "RL":
                eos = eosrl
            elif A == "UD":
                eos = eosud
            eo_sequences = eos[0][i]
            eo_lengths = eos[1][i]
            for B in ADJ[A]:
                drs = []
                if (A == "FB" and B == "UD"):
                    drs = drud_eofb
                elif(A == "FB" and B == "RL"):
                    drs = drrl_eofb
                elif(A == "RL" and B == "UD"):
                    drs = drud_eorl
                elif(A == "RL" and B == "FB"):
                    drs = drfb_eorl
                elif(A == "UD" and B == "FB"):
                    drs = drfb_eoud
                elif(A == "UD" and B == "RL"):
                    drs = drrl_eoud
                dr_sequences = drs[0][i]
                dr_lengths = drs[1][i]
                best_cost = None
                best_witness = {}
                support = 0
                for j, eo_sequence in enumerate(eo_sequences):
                    drs_from_eo_sequences = dr_sequences[j]
                    for k, dr in enumerate(drs_from_eo_sequences):
                        cost = eo_lengths[j] + dr_lengths[j][k]
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_witness = { #save the first best witness found.
                                "eo_len": eo_lengths[j], "dr_len": dr_lengths[j][k],
                                "eo_seq": eo_sequence, "dr_seq": dr
                            }
                            support = 1
                        elif cost == best_cost:
                            support += 1
                if best_cost is not None:
                    results.append({ #store the details of the shortest dr on each axis pairing.
                        "pair": f"EO_{A}->DR_{B}",
                        "score": best_cost,
                        "eo_len": best_witness["eo_len"],
                        "dr_len": best_witness["dr_len"],
                        "witness": best_witness,
                        "support": support
                    })
        if not results:
            return None
        results.sort(key=lambda r: (r["score"], r["eo_len"], -r["support"], r["dr_len"], r["pair"]))
        #tiebreaking scheme
        best = results[0]
        second = results[1]["score"] if len(results) > 1 else best["score"]
        gap = second - best["score"]
        row = {
            "scramble": scrambles[i],
            "label": best["pair"],
            "score": best["score"],
            "gap": gap,
            "eo_len": best["eo_len"],
            "dr_len": best["dr_len"],
            "support": best["support"],
            "witness_eo_seq": best["witness"].get("eo_seq", ""),
            "witness_dr_seq": best["witness"].get("dr_seq", ""),
            "witness_eo_len": best["witness"].get("eo_len", ""),
            "witness_dr_len": best["witness"].get("dr_len", "")
            #"all_pairs": results  # optional: keep for analysis
            }
        y[i] = row
    fields = ["scramble","label","score","gap","eo_len","dr_len","support","witness_eo_seq","witness_dr_seq","witness_eo_len","witness_dr_len"]
    with open("eodr_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in y:
            w.writerow(r)
    print(f"Wrote {S} rows to eodr_data.csv")    





    #make a list of answers e.g. drud from eofb.
    




    
    #From this scramble, are the short EOs on axis 1 or axis 2 closer to domino?

    #get 10 shortest EOs on each axis
    #for each EO, compute moves to DR + EO length
    #a human solver checks both axes once at EO so it's not much help to predict which is better once at EO.
    #It would be nice if there was correlation between scramble features and eo-axes' DR distances (both axes). 
    #but it's more reasonable that from a scrambled cube, there is correlation between 
    # eodr_axis features and eodr_axis distance.
    #So, I need to come up with the EODR_axis features (creativity) and the EODR_axis_ground_truth with careful code (will involve setting thresholds and whatnot).

if __name__ == "__main__":
    main()