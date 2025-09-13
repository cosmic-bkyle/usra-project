# Rubik's Cube USRA Project

This is a research‑oriented codebase exploring machine learning techniques to try to improve human performance in the [Fewest Moves Challenge](https://www.speedsolving.com/wiki/index.php?title=Fewest_Moves_Challenge) for the standard Rubik’s Cube. 

---
## Source Code

**[state.py](https://github.com/cosmic-bkyle/usra-project/blob/main/dr_to_solved/state.py)** provides an object-oriented cube state representation after user-input sequences of moves from the [Rubik's Cube Group](https://en.wikipedia.org/wiki/Rubik%27s_Cube_group). 

**[helpers.py](https://github.com/cosmic-bkyle/usra-project/blob/main/dr_to_solved/helpers.py)** contains helper functions to generate random elements of of the [domino subgroup](https://www.speedsolving.com/wiki/index.php/Domino_Reduction), portray adjacent pieces of a cube's permutation as a bipartite graph, and more. With the help of [nissy](https://nissy.tronto.net/), one million random domino scrambles along with their optimal solution lengths can be found in **[labelled_drs.parquet](https://github.com/cosmic-bkyle/usra-project/blob/main/dr_to_solved/labelled_drs.parquet)**.

**[learn_score.py](https://github.com/cosmic-bkyle/usra-project/blob/main/dr_to_solved/learn_score.py)** allows users to input their domino scramble and prints the linear model's guess of the distance from solved. As of 2025‑07‑05, the features are various types of blocks present, the corner solution length, and the "htr subset". 

The modules assume pre-installation of nissy.

---
## Research Journal
See **[journal.pdf](./journal.pdf)** for full notes. Report of findings is being written as of Sept 2025.

| Date           | Description                                                                     |
| -------------- | -------------------------------------------------------------------------------|
| 2025‑05‑20 | Clarified high-level goals & HTR subset stats task                                 |
| 2025‑05‑27 | Designed graph‑based blockiness feature                                            |
| 2025‑06‑03 | Implemented OOP cube, generated  dataset, fitted Lasso                      |