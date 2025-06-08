# Rubik's Cube USRA Project

This is a research‑oriented codebase exploring machine learning techniques to try to improve human performance in the [Fewest Moves Challenge](https://www.speedsolving.com/wiki/index.php?title=Fewest_Moves_Challenge) for the standard Rubik’s Cube. The project is underway for the summer of 2025!

---
## Source Code

**[state.py](https://github.com/cosmic-bkyle/usra-project/blob/main/state.py)** provides an object-oriented cube state representation after user-input sequences of moves from the [Rubik's Cube Group](https://en.wikipedia.org/wiki/Rubik%27s_Cube_group). 

**[helpers.py](https://github.com/cosmic-bkyle/usra-project/blob/main/helpers.py)** contains helper functions to generate random elements of certain subsets of the [domino subgroup](https://www.speedsolving.com/wiki/index.php/Domino_Reduction), portray adjacent pieces of a cube's permutation as a bipartite graph, and more. With the help of [nissy](https://nissy.tronto.net/), over 100,000 random domino scrambles along with their optimal solution lengths can be found in **[scrambles.csv](https://github.com/cosmic-bkyle/usra-project/blob/main/scrambles.csv)**.

**[regression.py](https://github.com/cosmic-bkyle/usra-project/blob/main/regression.py)** applies linear regression to to learn weights of engineered features. As of 2025‑06‑05, the features are the binary indicators of whether a corner and edge are touching in a domino scramble, and the target variable is the scramble's optimal solution length. The module assumes pre-installation of nissy.

---
## Research Journal
See **[journal.pdf](./journal.pdf)** for full notes

| Date           | Highlights                                                                     |
| -------------- | -------------------------------------------------------------------------------|
| 2025‑05‑20 | Clarified high-level goals & HTR subset stats task                                 |
| 2025‑05‑27 | Designed graph‑based blockiness feature                                            |
| 2025‑06‑03 | Implemented OOP cube, generated 100k DR dataset, fitted Lasso                      |