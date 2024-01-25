from torch.optim import SGD
from ChurnModel import ChurnModel
from torch.nn import Unflatten, MSELoss
from torch.utils.data import DataLoader
from BankChurnDataset import BankChurnDataset

dataset = BankChurnDataset("data/train.csv")
loader = DataLoader(dataset, 64)

model = ChurnModel()

loss = MSELoss()
optim = SGD(model.parameters(), 1e-9)

EPOCHS = 3
SIZE = len(loader.dataset)

model.train()


for i in range(EPOCHS):
    print(f"=== Running EPOCH {i+1}/{EPOCHS} ===\n")

    for idx, (inp, target) in enumerate(loader):
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
    out = model(inp)

    data.loc[len(data)] = [id, out]


data.to_csv("submission.csv")
