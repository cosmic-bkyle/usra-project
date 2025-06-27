'''
learn a final scoring function for a domino reduction using linear regression
'''
import pandas as pd, numpy as np, joblib, json, pathlib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, ElasticNet, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import csv
from sklearn.linear_model import LinearRegression
import dr_to_solved.helpers as helpers
import dr_to_solved.state as state

SUBSET_MEANS = {
                '0c0 0e': 9.242,
                '0c0 2e': 11.728, 
                '0c0 4e': 12.261, 
                '0c0 6e': 12.89, 
                '0c0 8e': 12.723, 
                '4a1 0e': 12.999, 
                '4a1 2e': 12.56, 
                '4a1 4e': 12.253, 
                '4b2 0e': 12.878, 
                '4b2 2e': 12.653, 
                '4b2 4e': 12.774, 
                '4a2 0e': 13.958, 
                '4a2 2e': 13.639, 
                '4a2 4e': 13.037, 
                '2c3 0e': 13.415, 
                '2c3 2e': 12.561, 
                '2c3 4e': 12.917, 
                '2c3 6e': 13.196, 
                '2c3 8e': 13.519, 
                '4c3 0e': 13.311, 
                '4c3 2e': 13.442, 
                '4c3 4e': 13.538, 
                '0c3 0e': 13.335, 
                '0c3 2e': 14.31, 
                '0c3 4e': 14.694, 
                '0c3 6e': 14.239, 
                '0c3 8e': 14.14, 
                '2c4 0e': 13.246, 
                '2c4 2e': 13.534, 
                '2c4 4e': 13.675, 
                '2c4 6e': 13.737, 
                '2c4 8e': 14.025, 
                '0c4 0e': 12.659, 
                '0c4 2e': 13.513, 
                '0c4 4e': 13.513, 
                '0c4 6e': 13.888, 
                '0c4 8e': 14.16, 
                '4c4 0e': 14.496, 
                '4c4 2e': 14.496, 
                '4c4 4e': 14.386, 
                '2c5 0e': 14.386, 
                '2c5 2e': 14.201, 
                '2c5 4e': 14.247, 
                '2c5 6e': 14.415, 
                '2c5 8e': 14.523, 
                '4c5 0e': 14.523, 
                '4c5 2e': 14.293, 
                '4c5 4e': 14.293}

def learn_and_print_weights():

    df = pd.read_parquet("labelled_drs.parquet")
    global_soln_mean = 13.580451
    global_corners_mean = 8.285799
    df["subset_mean"] = df["subset"].map(SUBSET_MEANS) #add column for subset mean
    df["subset_dev"] = df["subset_mean"] - global_soln_mean # how good the subset is
    df["corner_dev"] = df["corners"] - global_corners_mean   #how good the corner soln is

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=53, stratify=df["subset"])

    predictor_cols = ["subset_dev", "corner_dev", "pairs_a", "pairs_b", "lines", "good_L", "bad_L"]

    X_train = train_df[predictor_cols].values
    y_train = train_df["soln"].values

    X_test = test_df[predictor_cols].values
    y_test = test_df["soln"].values

    pipe = make_pipeline(
    StandardScaler(with_mean=False),   
    LassoCV(alphas=np.logspace(-3, 1, 30),   
             cv=5,
             random_state=42,
             max_iter=5000 )
    )         
    pipe.fit(X_train, y_train)
    lasso = pipe.named_steps["lassocv"] #get lasso model from pipeline

    intercept = lasso.intercept_
    coefs = dict(zip(predictor_cols, lasso.coef_))

    print(intercept)
    print(coefs)
    test_predictions = pipe.predict(X_test)
    rounded_test_preds = np.round(test_predictions)

    test_mae_rounded = mean_absolute_error(y_test, rounded_test_preds)
    #
    test_mae_unrounded = mean_absolute_error(y_test, test_predictions) #I checked and mean error is pretty much 0 as desired.

    print("Test MAE (unrounded):", test_mae_unrounded)
    print("Test MAE (rounded):", test_mae_rounded)

def guess(dr):
    ''' guess input dr's optimal solution length based on weights learned in learn_score'''

    soln = helpers.get_solns([dr, "R"])[0]
    corners = helpers.get_corner_solns([dr, "R"])[0]
    subset = helpers.get_subsets([dr])[0]
    cube = state.State()
    cube.apply(dr)
    a, b = cube.amt_pairs()
    lines = cube.amt_lines()
    goodL, badL = cube.amt_Ls()
    return round(13.885504045506273 + (SUBSET_MEANS[subset] - 13.580451) *0.48869033035039605 + (corners - 8.285799) * 0.19645535011114668 + a * (-0.16581759596099624) + b * (-0.16498669243675892) + goodL * (-0.01745540869875801) + badL * (0.050623909144940166) + lines * (0.03215225699166824))
    
def main():
    #learn_and_print_weights()

    scramble = helpers.get_dr_scrambles(1)[0]
    print(scramble)
    print(guess(scramble))
    print(helpers.get_solns([scramble, "U"]))


if __name__ == "__main__":
    main()
    

    #main()