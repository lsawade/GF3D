from builtins import list
from .source import CMTSOLUTION
from typing import List, Iterable
from .source import CMTSOLUTION


class CMTCatalog(List[CMTSOLUTION]):

    def __init__(self, cmts: Iterable[CMTSOLUTION]):
        super().__init__(cmts)

    # def sort(self, key=lambda x: (x.origin_time), reverse=False):
    #     self = sorted(self, key=key, reverse=reverse)

    @property
    def latitudes(self):
        return [cmt.latitude for cmt in self]

    @property
    def latitudes(self):
        return [cmt.latitude for cmt in self]
