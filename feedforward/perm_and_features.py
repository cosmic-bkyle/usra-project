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



def perm_one_hot(st: state.State) -> torch.Tensor:
    """
    """
    CORNERS, EDGES = 8, 12
    vec = torch.zeros(208, dtype=torch.float32)

    cperm, _ = st.corners
    for pos, cid in enumerate(cperm):
        vec[pos * CORNERS + cid] = 1.0

    eperm, _ = st.edges
    offset = CORNERS * CORNERS           
    for pos, eid in enumerate(eperm):
        vec[offset + pos * EDGES + eid] = 1.0
    return vec

class DominoDataset(Dataset):
    def __init__(self, path, split="train"):
        df = pd.read_parquet(path)

        n   = len(df)
        lo, hi = {"train": (0,.8), "val": (.8,.9), "test": (.9,1)}[split]
        df = df.iloc[int(lo*n): int(hi*n)].reset_index(drop=True)

        self.scrambles = df.scramble.tolist()
        self.soln      = df.soln.astype(np.float32).to_numpy()
        self.subset_id = df.subset_id.astype(np.int64).to_numpy()    # already 0…S‑1
        self.cornerLen = df.corners.astype(np.float32).to_numpy() # optimal corner len

    def __len__(self): return len(self.scrambles)

    
    def __getitem__(self, idx):
        st = state.State(); st.apply(self.scrambles[idx])

        blocky = torch.from_numpy(engineered_features(st))

        corner_dev = torch.tensor(self.cornerLen[idx]).unsqueeze(0)

        perm_vec = perm_one_hot(st)

        subset_id = torch.tensor(self.subset_id[idx])

        x_num = torch.cat([blocky, corner_dev])           #   6‑D
        return x_num, perm_vec, subset_id, torch.tensor(self.soln[idx])
class Net(nn.Module):
    def __init__(self, S, emb_dim=8, perm_dim=32):
        """
        S ........ # of HTR subsets (=48 in your data)
        emb_dim .. size of subset embedding
        perm_dim . compressed dimension for permutation signal
        """
        super().__init__()

        self.sub_emb   = nn.Embedding(S, emb_dim)

        self.perm_fc   = nn.Sequential(
            nn.Linear(208, perm_dim),
            nn.ReLU())

        in_dim = 6            + perm_dim + emb_dim  
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x_num, perm_vec, subset_id):
        z = torch.cat([ x_num,
                        self.perm_fc(perm_vec),
                        self.sub_emb(subset_id)], dim=-1)
        return self.mlp(z).squeeze(-1)
    
@torch.no_grad()
def mae(model, loader, device="cpu"):
    err, n = 0.0, 0
    for x_num, perm, sid, y in loader:
        x_num, perm, sid, y = (t.to(device) for t in (x_num, perm, sid, y))
        err += (model(x_num, perm, sid) - y).abs().sum().item()
        n   += len(y)
    return err/n
def train_one(parquet="dr_to_solved/labelled_drs.parquet",
              epochs=8, bs=2048, device="cpu"):

    tr_loader = DataLoader(DominoDataset(parquet, "train"),
                           batch_size=bs, shuffle=True,
                           num_workers=4, pin_memory=True)
    va_loader = DataLoader(DominoDataset(parquet, "val"),
                           batch_size=4096)

    net  = Net(48).to(device)
    opt  = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss()
    for ep in range(epochs):
        net.train()
        running, seen = 0., 0

        for x_num, perm, sid, y in tr_loader:
            x_num, perm, sid, y = (t.to(device) for t in (x_num, perm, sid, y))

            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x_num, perm, sid), y)
            loss.backward()
            opt.step()

            running += loss.item() * len(y)
            seen    += len(y)

        val_mae = mae(net, va_loader, device)
        print(f"epoch {ep:02d} | train‑loss {running/seen:.4f} | val‑MAE {val_mae:.4f}")
    torch.save(net.state_dict(), "oracle_perm_and_features.pth")
    return net

def main():
    parquet = "dr_to_solved/labelled_drs_with_id.parquet"
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print("using", device)

    net = train_one(parquet, epochs=8, bs=2048, device=device)

    te_loader = DataLoader(DominoDataset(parquet, "test"),
                           batch_size=4096, num_workers=4, pin_memory=True)
    print("TEST‑MAE", mae(net, te_loader, device))

if __name__ == "__main__":
    main()
    