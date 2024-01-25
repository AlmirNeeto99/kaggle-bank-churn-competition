import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset


class BankChurnDataset(Dataset):
    data: pd.DataFrame = None

    def __init__(self, path) -> None:
        super().__init__()
        self.data = pd.read_csv(path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tensor:
        pass
