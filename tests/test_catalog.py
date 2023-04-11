import pytest
import math
from obspy import UTCDateTime
from gf3d.source import CMTSOLUTION
from gf3d.catalog.cmt import CMTCatalog


cmtfile_1 = """
PDEW2015  4 16 18  7 43.00  35.1500   26.8300  20.0 0.0 6.0 CRETE, GREECE
event name:     201504161807A
time shift:      5.0000
half duration:   3.0000
latitude:       35.0000
longitude:      26.0000
depth:          24.0000
Mrr:       5.000000e+24
Mtt:      -1.000000e+25
Mpp:       6.000000e+24
Mrt:      -3.000000e+24
Mrp:       1.000000e+25
Mtp:      -3.000000e+24
"""

cmtfile_1_pert = """
PDEW2015  4 16 18  7 43.00  35.1500   26.8300  20.0 0.0 6.0 CRETE, GREECE
event name:     201504161807A
time shift:      4.0000
half duration:   2.0000
latitude:       34.0000
longitude:      25.0000
depth:          23.0000
Mrr:       4.000000e+24
Mtt:      -2.000000e+25
Mpp:       5.000000e+24
Mrt:      -4.000000e+24
Mrp:       0.000000e+25
Mtp:      -4.000000e+24
"""

cmtfile_2 = """
PDEW2016  4 16 18  7 43.00  35.1500   26.8300  20.0 0.0 6.0 CRETE, GREECE
event name:     201604161807A
time shift:      4.0000
half duration:   2.1000
latitude:       35.5000
longitude:      26.9000
depth:          15.0000
Mrr:       5.480000e+23
Mtt:      -1.230000e+24
Mpp:       6.850000e+25
Mrt:      -3.620000e+25
Mrp:       1.800000e+25
Mtp:      -3.050000e+23
"""

cmtfile_3 = """
PDEW2015  4 16 18  7 43.00  35.1500   26.8300  20.0 0.0 6.0 CRETE, GREECE
event name:     201504161807A
time shift:      5.0000
half duration:   3.0000
latitude:       35.0000
longitude:      26.0000
depth:          24.0000
Mrr:       5.000000e+24
Mtt:      -1.000000e+25
Mpp:       6.000000e+24
Mrt:      -3.000000e+24
Mrp:       1.000000e+25
Mtp:      -3.000000e+24
"""


@pytest.fixture
def cmt_tuple():
    return CMTSOLUTION.read(cmtfile_1), CMTSOLUTION.read(cmtfile_1_pert), CMTSOLUTION.read(cmtfile_2), CMTSOLUTION.read(cmtfile_3)


def test_sort(cmt_tuple):

    cat = CMTCatalog(cmt_tuple)

    cat.sort()

    # Since cmt 0 and cmt 3 are the same the order of them in the middle
    # should not affect the comparison
    assert (CMTCatalog([cmt_tuple[1], cmt_tuple[0],
            cmt_tuple[3], cmt_tuple[2]]) == cat)
    assert (CMTCatalog([cmt_tuple[1], cmt_tuple[3],
            cmt_tuple[0], cmt_tuple[2]]) == cat)
    assert ((
        CMTCatalog(
            [cmt_tuple[1], cmt_tuple[0], cmt_tuple[3], cmt_tuple[2]]) == cat)
            or (
        CMTCatalog(
            [cmt_tuple[1], cmt_tuple[3], cmt_tuple[0], cmt_tuple[2]]) == cat))
