from dr_to_solved import state
#need to debug my state for Corner orientation.


import pathlib
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from heapq import nsmallest, heappush, heappop
import pandas as pd



CORNERS = 8
EDGES = 12
DIM = 208
def encode_state(st):
    '''
    generate a 208-dimension feature vector of the current cubestate. e.g. if corner 3 is in position 1, 
    the vector has a 1 in the 1*8 + 3 = 11th dimension.

    '''

    feature_vec = np.zeros(DIM, dtype=np.float32)
    cperm, _ = st.corners  
    eperm, _ = st.edges

    for pos, corner_id in enumerate(cperm):
        feature_vec[pos * CORNERS + corner_id] = 1.0 

    offset = CORNERS * CORNERS
    for pos, edge_id in enumerate(eperm):
        feature_vec[offset + pos*EDGES + edge_id] = 1.0

    return feature_vec

class DominoDataset(Dataset):
    def __init__(self, data_path, split="train"):
        '''
        construct wrapped domino dataset from input csv or parquet.
        '''

        df = pd.read_parquet(pathlib.Path(data_path))
        scr = df['scramble'].tolist()
        dist = df['soln'].tolist()

        #80/10/10
        n     = len(scr) 
        idx, end = None, None
        if (split == "train"):
            idx = 0
            end = int(.8*n)
        elif (split == "val"):
            idx = int(.8*n)
            end = int(.9*n)
        elif (split == "test"):
            idx = int(.9*n)
            end= n
        
        self.scrambles = scr[idx:end]
        self.dists     = dist[idx:end]

    def __len__(self): 
        return len(self.scrambles)

    def __getitem__(self, i):
        ''' So that I can get a feature vector and target (v, k) using []
        '''
        st = state.State()
        st.apply(self.scrambles[i])
        x = torch.from_numpy(encode_state(st))  #torch.float32
        y = torch.tensor(self.dists[i], dtype=torch.float32)
        return x, y

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(208, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): 
        #define how data gets taken in. here x: (N, 208)
        return self.mlp(x).squeeze(-1)
    

@torch.no_grad()
def mae(model, loader, device="cpu"):
    '''
    compute mean actual error of a partially trained model on a dataloader.
    '''
    err, n = 0.0, 0
    for x,y in loader: #for each batch in the dataloader
        x,y = x.to(device), y.to(device)
        err += (model(x)-y).abs().sum().item() #sum abs errors over entire batch
        n   += len(y)
    return err/n
    
def train(parquet="dr_to_solved/labelled_drs.parquet", epochs=3, bs=1024, device="cpu"):
    train_dataloader = DataLoader(DominoDataset(parquet,"train"), bs, shuffle=True, #shuffle the epochs
                    num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(DominoDataset(parquet,"val"), batch_size =4096) #pick batch size

    net = Net().to(device) #model
    opt = torch.optim.SGD(net.parameters(), lr=1e-3) #library optimizer function another option is torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = loss_fn = nn.HuberLoss()  #library loss function

    for ep in range(epochs): #train in epochs, default 3.
        net.train() #train mode
        for x,y in train_dataloader:
            x,y = x.to(device), y.to(device)
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
        print(f"epoch {ep:02d}  val‑MAE {mae(net,val_dataloader,device):.3f}")
    torch.save(net.state_dict(), "oracle.pth")

    return net


def main():
    '''
    '''
    parquet = "dr_to_solved/labelled_drs.parquet"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device)

    net = train(parquet, epochs=8, bs=2048, device=device)

    test_dataloader = DataLoader(DominoDataset(parquet,"test"),
                             batch_size=4096,
                             num_workers=4,
                             pin_memory=True)
    
    net.eval() #inference mode
    test_mae = mae(net, test_dataloader, device)
    print("Test mae: " + str(test_mae))
    torch.save(net.state_dict(), "oracle_domino_v1.pth")

    #TODO: add more data points using short random walks, retrain the model
    #TODO: implement beam search to the solve dominoes.


if __name__ == "__main__":
    main()