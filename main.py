import torch
from torch.optim import SGD
from ChurnModel import ChurnModel
from torch.nn import Unflatten, MSELoss
from torch.utils.data import DataLoader
from BankChurnDataset import BankChurnDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using '{device}'")

dataset = BankChurnDataset("data/train.csv")
loader = DataLoader(dataset, 64)

model = ChurnModel().to(device)

loss = MSELoss()
optim = SGD(model.parameters(), 1e-9)

EPOCHS = 200
SIZE = len(loader.dataset)

model.train()


for i in range(EPOCHS):
    print(f"=== Running EPOCH {i+1}/{EPOCHS} ===\n")

    for idx, (inp, target) in enumerate(loader):
        inp, target = inp.to(device), target.to(device)
        pred = model(inp)

        unflat = Unflatten(0, (len(target), 1))
        u = unflat(target)
        error = loss(pred, u)
        error.backward()
        optim.step()
        optim.zero_grad()

        if idx % 100 == 0:
            l, current = error.item(), (idx + 1) * len(inp)
            print(f"LOSS: {l:>7f} - {current}/{SIZE}")

import pandas as pd

testD = BankChurnDataset("data/test.csv", True)
loader = DataLoader(testD, 1)


data = pd.DataFrame({"id": [], "Exited": []})

model.eval()
for idx, (id, inp) in enumerate(loader):
    id, inp = id.to(device), inp.to(device)
    out = model(inp)

    data.loc[len(data)] = [id.item(), out.item()]


data.to_csv("submission.csv", columns=["id", "Exited"], index=False)
