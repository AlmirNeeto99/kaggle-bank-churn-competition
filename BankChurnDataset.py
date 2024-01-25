import pandas as pd
from torch import tensor
from tiktoken import get_encoding
from torch.utils.data import Dataset


class BankChurnDataset(Dataset):
    data: pd.DataFrame = None

    def __init__(self, path) -> None:
        super().__init__()
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
        return tensor(self.data.iloc[index, self.indexes].to_list()), tensor(
            self.data.iloc[index, 13]
        )
