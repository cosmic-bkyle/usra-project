import torch
import torch.nn as nn
import torch.nn.functional as F
import cnn_helpers
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MaskedLinear(nn.Linear):
    """
    A linear layer whose weight matrix is permanently multiplied by a
    0/1 mask. Gradient updates on the zero entries are suppressed.
    """
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask)   #mask is a non-learnable tensor as part of the layer's state.
        with torch.no_grad():
            self.weight *= self.mask   #make sure initial zeroes stay zero

    def forward(self, input):
        # Weight • mask → in-place multiply keeps autograd happy
        return F.linear(input, self.weight * self.mask, self.bias)

def stickers_to_input_indices(sticker_ids):
    """
    given input sticker ids, return the list of all related indices in the onehot
    """
    ix = []
    for s in sticker_ids:
        ix.extend(range(s * 6, s * 6 + 6))
    return ix

def dict_to_mask(out_size, in_size, mapping, *, stickers=False):
    """
    mapping is a dictionary of form {out_node: [in_nodes]}
    where in_node is either a *sticker id* (if stickers=True) or the canonical numbering of a previous-layer node.
    """
    m = torch.zeros(out_size, in_size) #rows, columns. to be multiplied with weight matrix of the same shape. 
    # The weight matrix specifies the weights of input features / old neurons (cols) for each new neuron (row) at this level.
    for larger_feature, smaller_features in mapping.items():
        if stickers:    
            #each larger_feature is a piece/cubie, and smaller_features are the stickers on that cubie.
            for index in stickers_to_input_indices(smaller_features): #for each onehot index of the stickers on that cube
                m[larger_feature, index] = 1.0 #put a 1 in (cubie, index)
        else:
            for idx in smaller_features: 
                m[larger_feature, idx] = 1.0 #put a 1 for each subcomponent in this component's row.
    return m

'''
We need a mask for each hidden layer.
The first mask will be for the cubie layer, and will be of shape 16 cubes x 240 bits (onehot input size). 
The next mask is 24 pairs x 16 cubes
the next mask is 48 triples x 24 pairs
the next mask is 40 quadruples x 48 triples
'''

cube_mask = dict_to_mask(
    out_size = 16,
    in_size  = 240,
    mapping  = cnn_helpers.CUBE_TO_STICKERS,
    stickers = True
)
pair_mask = dict_to_mask(24, 16, cnn_helpers.PAIR_TO_CUBES)
triple_mask = dict_to_mask(48, 24, cnn_helpers.TRIPLE_TO_PAIRS)
quad_mask = dict_to_mask(40, 48, cnn_helpers.QUADRUPLE_TO_TRIPLES)





class DRPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.sticker_to_cube   = MaskedLinear(240, 16, cube_mask,   bias=True) #define layer
        self.cube_to_pair      = MaskedLinear(16,  24, pair_mask,   bias=True)
        self.pair_to_triple    = MaskedLinear(24,  48, triple_mask, bias=True)
        self.triple_to_quad    = MaskedLinear(48,  40, quad_mask,   bias=True)
        self.head              = nn.Linear(40, 1) #dense head, can change to multiple fully connected layers.
    def forward(self, x):
        """
        x is of shape (batch, 240), each row is an output of scramble_to_onehot

        Given input x, produce output by going through each defined layer and then a RELU
        """
        x = F.relu(self.sticker_to_cube(x))
        x = F.relu(self.cube_to_pair(x))
        x = F.relu(self.pair_to_triple(x))
        x = F.relu(self.triple_to_quad(x))
        x = self.head(x).squeeze(-1)  
        return x
    
class DRPredictor2(nn.Module):
    def __init__(self):
        super().__init__()
        self.sticker_to_cube   = MaskedLinear(240, 16, cube_mask,   bias=True) #define layer
        self.cube_to_pair      = MaskedLinear(16,  24, pair_mask,   bias=True)
        self.pair_to_triple    = MaskedLinear(24,  48, triple_mask, bias=True)
        self.triple_to_quad    = MaskedLinear(48,  40, quad_mask,   bias=True)
        self.head              = nn.Sequential(
            nn.Linear(40, 40), nn.ReLU(),   # 1st 40×40
            nn.Linear(40, 40), nn.ReLU(),   # 2nd 40×40
            nn.Linear(40, 40), nn.ReLU(),
            nn.Linear(40, 1)                # scalar output
        ) 
    def forward(self, x):
        """
        x is of shape (batch, 240), each row is an output of scramble_to_onehot

        Given input x, produce output by going through each defined layer and then a RELU
        """
        x = F.relu(self.sticker_to_cube(x))
        x = F.relu(self.cube_to_pair(x))
        x = F.relu(self.pair_to_triple(x))
        x = F.relu(self.triple_to_quad(x))
        x = self.head(x).squeeze(-1)  
        return x
    
class DRDataset(Dataset):
    def __init__(self, scrambles, targets):
        self.scrambles = scrambles
        self.targets   = np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.scrambles)

    def __getitem__(self, idx):
        x = cnn_helpers.scramble_to_onehot(self.scrambles[idx])
        y = self.targets[idx]
        return torch.from_numpy(x), torch.tensor(y)
    
def main1():
    parquet = "dr_to_solved/labelled_drs_with_id.parquet"
    df = pd.read_parquet(parquet)
    df = df[["scramble", "soln"]].dropna()
    df["soln"] = df["soln"].astype(np.float32)

    train_df, temp_df = train_test_split( # 70/30 split
        df, test_size=0.30, random_state=42, shuffle=True)
    val_df, test_df = train_test_split( #15/15 of the 30
        temp_df, test_size=0.50, random_state=42, shuffle=True)
    
    train_scrambles = train_df["scramble"].tolist()
    train_lengths   = train_df["soln"].tolist()

    val_scrambles   = val_df["scramble"].tolist()
    val_lengths     = val_df["soln"].tolist()

    test_scrambles  = test_df["scramble"].tolist()
    test_lengths    = test_df["soln"].tolist()

    print(f"▶  splits:  train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

    train_ds = DRDataset(train_scrambles, train_lengths)
    val_ds   = DRDataset(val_scrambles,   val_lengths)
    test_ds  = DRDataset(test_scrambles,  test_lengths)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=1024, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DRPredictor().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-4)
    lossfn = nn.MSELoss()

    for epoch in range(1, 20):
        model.train()
        for xb, yb in train_loader: #
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = lossfn(pred, yb)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        with torch.no_grad():
            mse_vals, mae_vals = [], []
            val_losses = []
            for xb, yb in val_loader: #don't update any weights in here, nor compute any gradients
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                mse_vals.append( lossfn(out, yb).item() )     # MSE, loss
                mae_vals.append( (out - yb).abs().mean().item() )  # MAE
        print(f"Epoch {epoch:02d} | "
          f"val MSE {np.mean(mse_vals):.4f} | "
          f"val RMSE {np.sqrt(np.mean(mse_vals)):.3f} | "
          f"val MAE {np.mean(mae_vals):.3f}")

    model.eval()
    with torch.no_grad(): #equivalent to val loop
        mse_tests, mae_tests = [], []
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out    = model(xb)
            mse_tests.append( lossfn(out, yb).item() )
            mae_tests.append( (out - yb).abs().mean().item() )
    test_mse  = np.mean(mse_tests)
    test_rmse = np.sqrt(test_mse)
    test_mae  = np.mean(mae_tests)
    print("\nFINAL  test MSE  {:.4f} | RMSE {:.3f} | MAE {:.3f}"
      .format(test_mse, test_rmse, test_mae))
    torch.save(model.state_dict(), "conv1.pth")

def main2():
    parquet = "dr_to_solved/labelled_drs_with_id.parquet"
    df = pd.read_parquet(parquet)
    df = df[["scramble", "soln"]].dropna()
    df["soln"] = df["soln"].astype(np.float32)

    train_df, temp_df = train_test_split( # 70/30 split
        df, test_size=0.30, random_state=42, shuffle=True)
    val_df, test_df = train_test_split( #15/15 of the 30
        temp_df, test_size=0.50, random_state=42, shuffle=True)
    
    train_scrambles = train_df["scramble"].tolist()
    train_lengths   = train_df["soln"].tolist()

    val_scrambles   = val_df["scramble"].tolist()
    val_lengths     = val_df["soln"].tolist()

    test_scrambles  = test_df["scramble"].tolist()
    test_lengths    = test_df["soln"].tolist()

    print(f"▶  splits:  train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

    train_ds = DRDataset(train_scrambles, train_lengths)
    val_ds   = DRDataset(val_scrambles,   val_lengths)
    test_ds  = DRDataset(test_scrambles,  test_lengths)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=1024, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DRPredictor2().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3) #learning rate
    lossfn = nn.MSELoss()

    for epoch in range(1, 20):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = lossfn(pred, yb)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        with torch.no_grad():
            mse_vals, mae_vals = [], []
            val_losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                mse_vals.append( lossfn(out, yb).item() )     # MSE, loss
                mae_vals.append( (out - yb).abs().mean().item() )  # MAE
        print(f"Epoch {epoch:02d} | "
          f"val MSE {np.mean(mse_vals):.4f} | "
          f"val RMSE {np.sqrt(np.mean(mse_vals)):.3f} | "
          f"val MAE {np.mean(mae_vals):.3f}")

    model.eval()
    with torch.no_grad():
        mse_tests, mae_tests = [], []
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out    = model(xb)
            mse_tests.append( lossfn(out, yb).item() )
            mae_tests.append( (out - yb).abs().mean().item() )
    test_mse  = np.mean(mse_tests)
    test_rmse = np.sqrt(test_mse)
    test_mae  = np.mean(mae_tests)
    print("\nFINAL  test MSE  {:.4f} | RMSE {:.3f} | MAE {:.3f}"
      .format(test_mse, test_rmse, test_mae))
    torch.save(model.state_dict(), "conv2.pth")



if __name__ == "__main__":
    print("First model: one head layer 40x1")
    main1()
    print("Second model: three fully connected 40x40 layers")
    main2()
