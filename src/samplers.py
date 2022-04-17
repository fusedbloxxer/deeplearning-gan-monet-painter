from typing import Generic, TypeVar, Iterator
from torch.utils.data import Sampler

T = TypeVar("T")


class InfSampler(Generic[T], Sampler[T]):
    def __init__(self, sampler: Sampler[T]):
        self.sampler = sampler

    def __iter__(self) -> Iterator[T]:
        while True:
            for sample in self.sampler:
                yield sample
