from dataclasses import dataclass
from collections import namedtuple
import numpy as np
import os


@dataclass
class TrainHistory(list):
    labels: list[str]
    
    
    def __post_init__(self):
        self._TrainHistoryTuple = namedtuple("_TrainHistoryTuple", self.labels)
        
    
    def append(self, entry):
        entry_tpl = self._TrainHistoryTuple(*entry)
        super().append(entry_tpl)
        
    
    @property
    def as_arrays(self):
        np_arrays = {
            lbl: np.array(arr)
            for lbl, arr in
                    zip(self.labels, zip(*self))
        }
        return np_arrays
    
    
    def save_arrays(self, fld):
        if os.path.exists(fld):
            raise FileExistsError(f'Cannot save here. Folder "{fld}" already exists.')
        else:
            os.makedirs(fld)
        for lbl, arr in self.as_arrays.items():
            path = os.path.join(fld, lbl + ".npy")
            with open(path, "wb") as f:
                np.save(f, arr)