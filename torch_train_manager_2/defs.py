from typing import Any, Dict, Iterator, Union
import torch.optim


Batch = Any
Prediction = Any
DataSource = Iterator[Batch]
LRSchedule = Any
ParamsOrGroups = Union[
    Iterator[torch.nn.Parameter], Dict[str, Iterator[torch.nn.Parameter]]
]
ScalarTensor = torch.Tensor
""" A torch tensor that is a scalar """
