import numpy as np
from .constants_solver import NGLLX, NGLLY, NGLLZ, MIDX, MIDY, MIDZ, NGNOD


def hex_nodes():
    """
    ! topology of the elements
    integer, dimension(NGNOD), intent(out):: iaddx, iaddy, iaddz

    ! define the topology of the hexahedral elements

    ! the topology of the nodes is described in UTILS/chunk_notes_scanned/numbering_convention_27_nodes.tif
    """

    if (NGNOD != 27):
        raise ValueError('Elements should have 27 control nodes')

    iaddx = np.zeros(27)
    iaddy = np.zeros(27)
    iaddz = np.zeros(27)

    # corner nodes

    iaddx[0] = 0
    iaddy[0] = 0
    iaddz[0] = 0

    iaddx[1] = 2
    iaddy[1] = 0
    iaddz[1] = 0

    iaddx[2] = 2
    iaddy[2] = 2
    iaddz[2] = 0

    iaddx[3] = 0
    iaddy[3] = 2
    iaddz[3] = 0

    iaddx[4] = 0
    iaddy[4] = 0
    iaddz[4] = 2

    iaddx[5] = 2
    iaddy[5] = 0
    iaddz[5] = 2

    iaddx[6] = 2
    iaddy[6] = 2
    iaddz[6] = 2

    iaddx[7] = 0
    iaddy[7] = 2
    iaddz[7] = 2

    # midside nodes(nodes located in the middle of an edge)

    iaddx[8] = 1
    iaddy[8] = 0
    iaddz[8] = 0

    iaddx[9] = 2
    iaddy[9] = 1
    iaddz[9] = 0

    iaddx[10] = 1
    iaddy[10] = 2
    iaddz[10] = 0

    iaddx[11] = 0
    iaddy[11] = 1
    iaddz[11] = 0

    iaddx[12] = 0
    iaddy[12] = 0
    iaddz[12] = 1

    iaddx[13] = 2
    iaddy[13] = 0
    iaddz[13] = 1

    iaddx[14] = 2
    iaddy[14] = 2
    iaddz[14] = 1

    iaddx[15] = 0
    iaddy[15] = 2
    iaddz[15] = 1

    iaddx[16] = 1
    iaddy[16] = 0
    iaddz[16] = 2

    iaddx[17] = 2
    iaddy[17] = 1
    iaddz[17] = 2

    iaddx[18] = 1
    iaddy[18] = 2
    iaddz[18] = 2

    iaddx[19] = 0
    iaddy[19] = 1
    iaddz[19] = 2

    # side center nodes(nodes located in the middle of a face)

    iaddx[20] = 1
    iaddy[20] = 1
    iaddz[20] = 0

    iaddx[21] = 1
    iaddy[21] = 0
    iaddz[21] = 1

    iaddx[22] = 2
    iaddy[22] = 1
    iaddz[22] = 1

    iaddx[23] = 1
    iaddy[23] = 2
    iaddz[23] = 1

    iaddx[24] = 0
    iaddy[24] = 1
    iaddz[24] = 1

    iaddx[25] = 1
    iaddy[25] = 1
    iaddz[25] = 2

    # center node(barycenter of the eight corners)

    iaddx[26] = 1
    iaddy[26] = 1
    iaddz[26] = 1

    return iaddx, iaddy, iaddz


def hex_nodes_anchor_ijk():
    """
    gets control point indices

    to get coordinates of control points(corners, midpoints) for an element ispec, they can be use as:
    do ia = 1, NGNOD
      iglob = ibool(anchor_iax(ia), anchor_iay(ia), anchor_iaz(ia), ispec)
      xelm(ia) = dble(xstore(iglob))
      yelm(ia) = dble(ystore(iglob))
      zelm(ia) = dble(zstore(iglob))
    enddo
    """

    # preset the achors
    anchor_iax, anchor_iay, anchor_iaz = np.zeros(
        NGNOD), np.zeros(NGNOD), np.zeros(NGNOD)

    # define topology of the control element
    iaddx, iaddy, iaddr = hex_nodes()

    # define(i, j, k) indices of the control/anchor points of the elements
    for ia in range(NGNOD):
        # control point index
        iax = 0
        if (iaddx[ia] == 0):
            iax = 1
        elif (iaddx[ia] == 1):
            iax = MIDX
        elif (iaddx[ia] == 2):
            iax = NGLLX
        else:
            raise ValueError('incorrect value of iaddx')

        anchor_iax[ia] = iax

        iay = 0
        if (iaddy[ia] == 0):
            iay = 1
        elif (iaddy[ia] == 1):
            iay = MIDY
        elif (iaddy[ia] == 2):
            iay = NGLLY
        else:
            raise ValueError('incorrect value of iaddy')

        anchor_iay[ia] = iay

        iaz = 0
        if (iaddr[ia] == 0):
            iaz = 1
        elif (iaddr[ia] == 1):
            iaz = MIDZ
        elif (iaddr[ia] == 2):
            iaz = NGLLZ
        else:
            raise ValueError('incorrect value of iaddz')

        anchor_iaz[ia] = iaz

    return anchor_iax, anchor_iay, anchor_iaz
