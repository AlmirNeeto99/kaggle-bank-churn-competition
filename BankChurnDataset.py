import pandas as pd
from torch import tensor, float32
from tiktoken import get_encoding
from torch.utils.data import Dataset


class BankChurnDataset(Dataset):
    data: pd.DataFrame = None

    def __init__(self, path, isTest=False) -> None:
        super().__init__()
        self.isTest = isTest
        self.data = pd.read_csv(path)
        self.indexes = list(range(3, 13))
        self.process()

    def _encode(self, row):
        row["Gender"] = 0 if row["Gender"] == "Male" else 1
        row["Geography"] = self.mapping[row["Geography"]][0]
        return row

    def process(self):
        un = self.data.Geography.unique()
        encoder = get_encoding("cl100k_base")
        self.geo = list(map(lambda y: encoder.encode(y), un))
        self.mapping = dict()
        for k, v in zip(un, self.geo):
            self.mapping[k] = v
        self.data = self.data.apply(self._encode, 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        if self.isTest:
            return (
                tensor(self.data.iloc[index, 0]),
                tensor(
                    self.data.iloc[index, self.indexes].to_list(), dtype=float32
                ),
            )

        return tensor(
            self.data.iloc[index, self.indexes].to_list(), dtype=float32
        ), tensor(self.data.iloc[index, 13], dtype=float32)
