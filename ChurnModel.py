from torch import nn, Tensor


class ChurnModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(10, 10240)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(10240, 160)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(160, 1)
        self.a3 = nn.Sigmoid()

    def forward(self, x) -> Tensor:
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        x = self.a3(x)

        return x
