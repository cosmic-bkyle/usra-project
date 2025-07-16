'''
first hidden layer represents all pair locations in a domino. A mask matrix mutes all of the "irrelevant" connections to the input layer. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from dr_to_solved import state
import pathlib
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from dr_to_solved import state

CORNERS = 8
EDGES = 12
DIM = 208
def encode_state(st):
    '''
    generate 208-dimension feature vector of the current cubestate. e.g. if corner 3 is in position 1, 
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

ALL_PAIRS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 9), (5, 10), (6, 11), (7, 8), (0, 1), (1, 2), (2, 3), (3, 0), (4, 8), (5, 9), (6, 10), (7, 11)]
#a and b pairs

class MaskedLinear(nn.Linear):
    """
    nn.Linear subclass
    weight matrix W is element‑wise multiplied by fixed mask (passed as input)
    before every forward() and every optimiser step.

    input mask must have the same shape as W: (out_features, in_features)
    it is sparse, with all pair-irrelevant elements as 0.

    """

    def __init__(self, in_features, out_features, bias=True, *, mask):
        super().__init__(in_features, out_features, bias=bias) #process input and output size.

        #register mask as a buffer
        self.register_buffer("mask", mask) #tensors that are part of your model's state but are not intended to be updated 
        with torch.no_grad():
            self.weight *= self.mask #element-wise multiplication

    def forward(self, input):
        if self.weight.grad is not None:
            self.weight.grad *= self.mask #mask gradients
        W = self.weight * self.mask
        return F.linear(input, W, self.bias)
    

class PairLayer(nn.Module):
    """
    Constructor requires a pair_mask matrix.
    208d onehot -> 16 pairs -> fully connected linear layers -> output
    """
    def __init__(self, pair_mask):     
        super().__init__()
        self.pair_layer = MaskedLinear(
            in_features=208,
            out_features=pair_mask.size(0),
            bias=True,
            mask=pair_mask)

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1))

    def forward(self, x): #x is shape(batch, 208)
        '''how to turn an input tensor into the output tensor'''
        pairs = self.pair_layer(x) #masked connectivity
        return self.head(pairs).squeeze(-1)   #(batch,)

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
    

def cols_for_position(pos_id, is_corner):
    """
    make list of column indices for a position (eight for corners, 12 for edges).
    An edge id in the input to this method is from 0-11.
    """
    if is_corner:               
        base = pos_id * 8
        return list(range(base, base+8))
    else:                       
        base = 64 + pos_id * 12 
        return list(range(base, base+12))
    
def train(net, parquet="dr_to_solved/labelled_drs.parquet", epochs=3, bs=1024, device="cpu"):
    train_dataloader = DataLoader(DominoDataset(parquet,"train"), bs, shuffle=True, #shuffle the epochs
                    num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(DominoDataset(parquet,"val"), batch_size =4096) #pick batch size


    opt = optimizer = torch.optim.SGD(net.parameters(), lr=1e-3) #library optimizer function another option is torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = loss_fn = nn.HuberLoss()  #library loss function

    for ep in range(epochs):
        net.train()
        running_loss, seen = 0., 0
        for x,y in train_dataloader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * len(y)
            seen         += len(y)

        net.eval()
        val_mae = mae(net, val_dataloader, device)
        net.train()

        print(f"epoch {ep:02d} | train‑loss {running_loss/seen:.4f} | "
            f"val‑MAE {val_mae:.4f}")
    torch.save(net.state_dict(), "oracle.pth")

    return net

def main():
    parquet = "dr_to_solved/labelled_drs.parquet"
    
    mask = torch.zeros(16, 208)
    for row, (cpos, epos) in enumerate(ALL_PAIRS): #16 tuples
        cols = cols_for_position(cpos, True) + cols_for_position(epos, False)
        mask[row, cols] = 1.0 #add twenty 1s to the matrix in current row.

    net = PairLayer(mask)
    net = train(net, parquet, 15, bs=2048, device="cpu")


    test_dataloader = DataLoader(DominoDataset(parquet,"test"),
                             batch_size=4096,
                             num_workers=4,
                             pin_memory=True)
    
    net.eval() #inference mode
    test_mae = mae(net, test_dataloader, "cpu")
    print("Test mae: " + str(test_mae))
    torch.save(net.state_dict(), "pair_mask.pth")
    

def decode_column(col):
    '''decode the meaning of a column in a one-hot.'''
    if col < 64: #Corner relation                 
        corner_pos  = col//8   
        corner_id   = col%8    
        return f"corner {corner_id} in position {corner_pos}"
    else:                        
        col -= 64 #edge relation 
        edge_pos = col//12  
        edge_id = col%12 
        return f"edge {edge_id} in positon {edge_pos}"

if __name__ == "__main__":
    #main() #train model, print and save results

    mask = torch.zeros(16, 208) #for each pair, which 
    for row, (cpos, epos) in enumerate(ALL_PAIRS): #16 tuples
        cols = cols_for_position(cpos, True) + cols_for_position(epos, False)
        mask[row, cols] = 1.0 #add twenty 1s to the matrix in current row.
    #init mask

    net = PairLayer(mask)
    net.load_state_dict(torch.load("pair_mask.pth", map_location="cpu"))
    net.eval()
    W = net.pair_layer.weight * net.pair_layer.mask 


    pair_idx = 4  
    row = W[pair_idx].detach().cpu().numpy() #get the 206-dim onehot vector that this pair sees.

    #largest positive and negative links
    top = row.argsort()[-30:][::-1] #five strongest pos weights
    bot = row.argsort()[:20]   #five most negative

    print("Pair " + str(pair_idx))
    for c in top:
        print(f"  {row[c]:.3f}  from  {decode_column(c)}")
    for c in bot:
        print(f"  {row[c]:.3f}  from  {decode_column(c)}")
    '''
    Pair 3:
  +0.045  from  corner ID7 @ pos3
  +0.031  from  corner ID2 @ pos3
  +0.030  from  edge ID5 @ pos3
  +0.021  from  edge ID9 @ pos3
  +0.017  from  corner ID3 @ pos3
  -0.068  from  edge ID7 @ pos3
  -0.067  from  edge ID1 @ pos3
  -0.067  from  edge ID0 @ pos3
  -0.062  from  edge ID8 @ pos3
  -0.057  from  edge ID4 @ pos3

  Pair 4:
  +0.081  from  corner ID4 @ pos4
  +0.073  from  edge ID2 @ pos9
  +0.061  from  corner ID0 @ pos4
  +0.058  from  corner ID5 @ pos4
  +0.050  from  edge ID4 @ pos9
  -0.041  from  corner ID3 @ pos4
  -0.015  from  edge ID6 @ pos9
  -0.013  from  edge ID7 @ pos9
  -0.013  from  edge ID8 @ pos9
  -0.010  from  corner ID7 @ pos4
    
    
    '''