from dr_to_solved import state
#need to debug my state for Corner orientation.


import pathlib
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from heapq import nsmallest, heappush, heappop
import pandas as pd

def engineered_features(st: state.State) -> np.ndarray:
    """Return the 5‑D vector  [pair_a, pair_b, goodL, badL, lines]."""
    pair_a, pair_b     = st.amt_pairs()
    goodL,  badL       = st.amt_Ls()
    lines              = st.amt_lines()
    return np.asarray([pair_a, pair_b, goodL, badL, lines], dtype=np.float32)


class DominoDataset(Dataset):
    def __init__(self, path, split="train"):
        df = pd.read_parquet(path)

        n   = len(df)
        lo, hi = {"train": (0,.8), "val": (.8,.9), "test": (.9,1)}[split]
        df = df.iloc[int(lo*n): int(hi*n)].reset_index(drop=True)

        self.scrambles = df.scramble.tolist()
        self.soln      = df.soln.astype(np.float32).to_numpy()
        self.subset_id = df.subset_id.astype(np.int64).to_numpy()   
        self.cornerLen = df.corners.astype(np.float32).to_numpy() #

    def __len__(self): return len(self.scrambles)

    def __getitem__(self, idx):
        st = state.State();  st.apply(self.scrambles[idx])

        feat = torch.from_numpy(engineered_features(st))        
        corner_dev = torch.tensor(self.cornerLen[idx])

        subset_id  = torch.tensor(self.subset_id[idx])

        x = torch.cat([feat, corner_dev.unsqueeze(0)])          
        y = torch.tensor(self.soln[idx])
        return x, subset_id, y

class Net1(nn.Module):
    def __init__(self, S, emb_dim=8):
        super().__init__()
        self.sub_emb = nn.Embedding(S, emb_dim)  

        inp = 6 + emb_dim                         
        self.mlp = nn.Sequential( #container module
            nn.Linear(inp, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x_num, subset_id):
        z = torch.cat([x_num, self.sub_emb(subset_id)], dim=-1)
        return self.mlp(z).squeeze(-1)

class Net2(nn.Module):
    def __init__(self, S, emb_dim=8):
        super().__init__()
        self.sub_emb = nn.Embedding(S, emb_dim)  

        inp = 6 + emb_dim                         
        self.mlp = nn.Sequential( #container module
            nn.Linear(inp, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x_num, subset_id):
        z = torch.cat([x_num, self.sub_emb(subset_id)], dim=-1)
        return self.mlp(z).squeeze(-1)
    

class Net3(nn.Module):
    def __init__(self, S, emb_dim=8, hid=256):
        super().__init__()
        self.sub_emb = nn.Embedding(S, emb_dim)
        inp = 6 + emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(),
            nn.Linear(hid, hid//2), nn.ReLU(),
            nn.Linear(hid//2, hid//4), nn.ReLU(),
            nn.Linear(hid//4, 1)
        )

    def forward(self, x_num, subset_id):
        z = torch.cat([x_num, self.sub_emb(subset_id)], dim=-1)
        return self.mlp(z).squeeze(-1)


@torch.no_grad()
def mae(model, loader, device="cpu"):
    '''
    compute mean actual error of a partially trained model on a dataloader.
    '''
    err, n = 0.0, 0
    for x, sid, y in loader: #for each batch in the dataloader
        x, sid, y = x.to(device), sid.to(device), y.to(device)
        err += (model(x, sid)-y).abs().sum().item() #sum abs errors over entire batch
        n   += len(y)
    return err/n   

def train_one(parquet="dr_to_solved/labelled_drs.parquet",
              epochs=8, bs=2048, device="cpu"):

    tr_loader = DataLoader(DominoDataset(parquet, "train"),
                           batch_size=bs, shuffle=True,
                           num_workers=4, pin_memory=True)
    va_loader = DataLoader(DominoDataset(parquet, "val"),
                           batch_size=4096)

    net  = Net3(48).to(device) #changeable
    opt  = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss()

    for ep in range(epochs):
        net.train()
        running_loss, seen = 0., 0
        for x, sid, y in tr_loader:
            x, sid, y = x.to(device), sid.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x, sid), y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * len(y)
            seen         += len(y)

        val_mae = mae(net, va_loader, device)
        print(f"epoch {ep:02d} | train‑loss {running_loss/seen:.4f} | "
              f"val‑MAE {val_mae:.4f}")

    #torch.save(net.state_dict(), "oracle_features.pth")
    return net

def main():
    '''
    change between model architectures at line 117
    
    '''
    parquet = "dr_to_solved/labelled_drs_with_id.parquet"
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print("using", device)

    net = train_one(parquet, epochs=15, bs=2048, device=device)

    te_loader = DataLoader(DominoDataset(parquet, "test"),
                           batch_size=4096, num_workers=4, pin_memory=True)
    print("TEST‑MAE", mae(net, te_loader, device))


if __name__ == "__main__":
    main()