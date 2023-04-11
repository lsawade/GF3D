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
def cmt1():
    return CMTSOLUTION.read(cmtfile_1)


@pytest.fixture
def cmt1_pert():
    return CMTSOLUTION.read(cmtfile_1_pert)


@pytest.fixture
def cmt2():
    return CMTSOLUTION.read(cmtfile_2)


@pytest.fixture
def cmt3():
    return CMTSOLUTION.read(cmtfile_3)


def test_equal(cmt1, cmt2, cmt3):

    # 1 and 3 are the same
    assert cmt1 == cmt3

    # 1 and 3 differ from 2
    assert cmt2 != cmt1
    assert cmt2 != cmt3


def test_greater_equal(cmt1, cmt2, cmt3):

    # 1 and 3 are the same
    assert cmt1 >= cmt3

    assert (cmt1 > cmt3) == False

    # 1 and 3 differ from 2
    assert cmt2 >= cmt1
    assert cmt2 >= cmt3


def test_greater_than(cmt1, cmt2, cmt3):

    # 1 and 3 differ from 2
    assert cmt2 > cmt1
    assert cmt2 > cmt3


def test_sub(cmt1, cmt1_pert):

    # 1 and 3 differ from 2
    assert (cmt1 - cmt1_pert).origin_time == 0.0
    assert (cmt1 - cmt1_pert).eventname == cmt1.eventname
    assert (cmt1 - cmt1_pert).eventname == cmt1_pert.eventname
    assert (cmt1 - cmt1_pert).time_shift == 1.0
    assert (cmt1 - cmt1_pert).hdur == 1.0
    assert (cmt1 - cmt1_pert).latitude == 1.0
    assert (cmt1 - cmt1_pert).longitude == 1.0
    assert (cmt1 - cmt1_pert).depth == 1.0
    assert math.isclose((cmt1 - cmt1_pert).Mrr, 1.0e+24)
    assert math.isclose((cmt1 - cmt1_pert).Mtt, 1.0e+25)
    assert math.isclose((cmt1 - cmt1_pert).Mpp, 1.0e+24)
    assert math.isclose((cmt1 - cmt1_pert).Mrt, 1.0e+24)
    assert math.isclose((cmt1 - cmt1_pert).Mrp, 1.0e+25)
    assert math.isclose((cmt1 - cmt1_pert).Mtp, 1.0e+24)
