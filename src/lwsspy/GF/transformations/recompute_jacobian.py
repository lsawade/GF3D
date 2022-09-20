import numpy as np
from lwsspy.GF.constants import ZERO, HALF, ONE, TWO
from lwsspy.GF.constants_solver import NGNOD, NDIM


def recompute_jacobian(
        xelm, yelm, zelm, xi, eta, gamma, x, y, z,
        xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz):
    """
    use constants, only: NGNOD,NDIM,ZERO,HALF,ONE,TWO

    implicit none

    double precision,intent(inout) :: x,y,z,xix,xiy,xiz,etax,etay,etaz,gammax,gammay,gammaz
    double precision,intent(in) :: xi,eta,gamma

    ! coordinates of the control points of the surface element
    double precision,intent(in) :: xelm(NGNOD),yelm(NGNOD),zelm(NGNOD)

    ! 3D shape functions and their derivatives at receiver
    double precision :: shape3D(NGNOD)
    double precision :: dershape3D(NDIM,NGNOD)

    double precision :: l1xi,l2xi,l3xi
    double precision :: l1eta,l2eta,l3eta
    double precision :: l1gamma,l2gamma,l3gamma
    double precision :: l1pxi,l2pxi,l3pxi
    double precision :: l1peta,l2peta,l3peta
    double precision :: l1pgamma,l2pgamma,l3pgamma

    double precision :: xxi,yxi,zxi
    double precision :: xeta,yeta,zeta
    double precision :: xgamma,ygamma,zgamma
    double precision :: jacobian

    integer :: ia

    ! recompute Jacobian for any given (xi,eta,gamma) point
    ! not necessarily a GLL point
    """

    # check that the parameter file is correct
    if (NGNOD /= 27):
        raise ValueError('elements should have 27 control nodes')

    l1xi = HALF*xi*(xi-ONE)
    l2xi = ONE-xi**2
    l3xi = HALF*xi*(xi+ONE)

    l1pxi = xi-HALF
    l2pxi = -TWO*xi
    l3pxi = xi+HALF

    l1eta = HALF*eta*(eta-ONE)
    l2eta = ONE-eta**2
    l3eta = HALF*eta*(eta+ONE)

    l1peta = eta-HALF
    l2peta = -TWO*eta
    l3peta = eta+HALF

    l1gamma = HALF*gamma*(gamma-ONE)
    l2gamma = ONE-gamma**2
    l3gamma = HALF*gamma*(gamma+ONE)

    l1pgamma = gamma-HALF
    l2pgamma = -TWO*gamma
    l3pgamma = gamma+HALF

    shape3D = np.zeros(NGNOD)
    dershape3D = np.zeros(NDIM, NGNOD)

    # corner nodes
    shape3D[0] = l1xi*l1eta*l1gamma
    shape3D[1] = l3xi*l1eta*l1gamma
    shape3D[2] = l3xi*l3eta*l1gamma
    shape3D[3] = l1xi*l3eta*l1gamma
    shape3D[4] = l1xi*l1eta*l3gamma
    shape3D[5] = l3xi*l1eta*l3gamma
    shape3D[6] = l3xi*l3eta*l3gamma
    shape3D[7] = l1xi*l3eta*l3gamma

    dershape3D[0, 0] = l1pxi*l1eta*l1gamma
    dershape3D[0, 1] = l3pxi*l1eta*l1gamma
    dershape3D[0, 2] = l3pxi*l3eta*l1gamma
    dershape3D[0, 3] = l1pxi*l3eta*l1gamma
    dershape3D[0, 4] = l1pxi*l1eta*l3gamma
    dershape3D[0, 5] = l3pxi*l1eta*l3gamma
    dershape3D[0, 6] = l3pxi*l3eta*l3gamma
    dershape3D[0, 7] = l1pxi*l3eta*l3gamma

    dershape3D[1, 0] = l1xi*l1peta*l1gamma
    dershape3D[1, 1] = l3xi*l1peta*l1gamma
    dershape3D[1, 2] = l3xi*l3peta*l1gamma
    dershape3D[1, 3] = l1xi*l3peta*l1gamma
    dershape3D[1, 4] = l1xi*l1peta*l3gamma
    dershape3D[1, 5] = l3xi*l1peta*l3gamma
    dershape3D[1, 6] = l3xi*l3peta*l3gamma
    dershape3D[1, 7] = l1xi*l3peta*l3gamma

    dershape3D[2, 0] = l1xi*l1eta*l1pgamma
    dershape3D[2, 1] = l3xi*l1eta*l1pgamma
    dershape3D[2, 2] = l3xi*l3eta*l1pgamma
    dershape3D[2, 3] = l1xi*l3eta*l1pgamma
    dershape3D[2, 4] = l1xi*l1eta*l3pgamma
    dershape3D[2, 5] = l3xi*l1eta*l3pgamma
    dershape3D[2, 6] = l3xi*l3eta*l3pgamma
    dershape3D[2, 7] = l1xi*l3eta*l3pgamma

    # midside nodes
    shape3D[8] = l2xi*l1eta*l1gamma
    shape3D[9] = l3xi*l2eta*l1gamma
    shape3D[10] = l2xi*l3eta*l1gamma
    shape3D[11] = l1xi*l2eta*l1gamma
    shape3D[12] = l1xi*l1eta*l2gamma
    shape3D[13] = l3xi*l1eta*l2gamma
    shape3D[14] = l3xi*l3eta*l2gamma
    shape3D[15] = l1xi*l3eta*l2gamma
    shape3D[16] = l2xi*l1eta*l3gamma
    shape3D[17] = l3xi*l2eta*l3gamma
    shape3D[18] = l2xi*l3eta*l3gamma
    shape3D[19] = l1xi*l2eta*l3gamma

    dershape3D[0, 8] = l2pxi*l1eta*l1gamma
    dershape3D[0, 9] = l3pxi*l2eta*l1gamma
    dershape3D[0, 10] = l2pxi*l3eta*l1gamma
    dershape3D[0, 11] = l1pxi*l2eta*l1gamma
    dershape3D[0, 12] = l1pxi*l1eta*l2gamma
    dershape3D[0, 13] = l3pxi*l1eta*l2gamma
    dershape3D[0, 14] = l3pxi*l3eta*l2gamma
    dershape3D[0, 15] = l1pxi*l3eta*l2gamma
    dershape3D[0, 16] = l2pxi*l1eta*l3gamma
    dershape3D[0, 17] = l3pxi*l2eta*l3gamma
    dershape3D[0, 18] = l2pxi*l3eta*l3gamma
    dershape3D[0, 19] = l1pxi*l2eta*l3gamma

    dershape3D[1, 8] = l2xi*l1peta*l1gamma
    dershape3D[1, 9] = l3xi*l2peta*l1gamma
    dershape3D[1, 10] = l2xi*l3peta*l1gamma
    dershape3D[1, 11] = l1xi*l2peta*l1gamma
    dershape3D[1, 12] = l1xi*l1peta*l2gamma
    dershape3D[1, 13] = l3xi*l1peta*l2gamma
    dershape3D[1, 14] = l3xi*l3peta*l2gamma
    dershape3D[1, 15] = l1xi*l3peta*l2gamma
    dershape3D[1, 16] = l2xi*l1peta*l3gamma
    dershape3D[1, 17] = l3xi*l2peta*l3gamma
    dershape3D[1, 18] = l2xi*l3peta*l3gamma
    dershape3D[1, 19] = l1xi*l2peta*l3gamma

    dershape3D[2, 8] = l2xi*l1eta*l1pgamma
    dershape3D[2, 9] = l3xi*l2eta*l1pgamma
    dershape3D[2, 10] = l2xi*l3eta*l1pgamma
    dershape3D[2, 11] = l1xi*l2eta*l1pgamma
    dershape3D[2, 12] = l1xi*l1eta*l2pgamma
    dershape3D[2, 13] = l3xi*l1eta*l2pgamma
    dershape3D[2, 14] = l3xi*l3eta*l2pgamma
    dershape3D[2, 15] = l1xi*l3eta*l2pgamma
    dershape3D[2, 16] = l2xi*l1eta*l3pgamma
    dershape3D[2, 17] = l3xi*l2eta*l3pgamma
    dershape3D[2, 18] = l2xi*l3eta*l3pgamma
    dershape3D[2, 19] = l1xi*l2eta*l3pgamma

    # side center nodes
    shape3D[20] = l2xi*l2eta*l1gamma
    shape3D[21] = l2xi*l1eta*l2gamma
    shape3D[22] = l3xi*l2eta*l2gamma
    shape3D[23] = l2xi*l3eta*l2gamma
    shape3D[24] = l1xi*l2eta*l2gamma
    shape3D[25] = l2xi*l2eta*l3gamma

    dershape3D[0, 20] = l2pxi*l2eta*l1gamma
    dershape3D[0, 21] = l2pxi*l1eta*l2gamma
    dershape3D[0, 22] = l3pxi*l2eta*l2gamma
    dershape3D[0, 23] = l2pxi*l3eta*l2gamma
    dershape3D[0, 24] = l1pxi*l2eta*l2gamma
    dershape3D[0, 25] = l2pxi*l2eta*l3gamma

    dershape3D[1, 20] = l2xi*l2peta*l1gamma
    dershape3D[1, 21] = l2xi*l1peta*l2gamma
    dershape3D[1, 22] = l3xi*l2peta*l2gamma
    dershape3D[1, 23] = l2xi*l3peta*l2gamma
    dershape3D[1, 24] = l1xi*l2peta*l2gamma
    dershape3D[1, 25] = l2xi*l2peta*l3gamma

    dershape3D[2, 20] = l2xi*l2eta*l1pgamma
    dershape3D[2, 21] = l2xi*l1eta*l2pgamma
    dershape3D[2, 22] = l3xi*l2eta*l2pgamma
    dershape3D[2, 23] = l2xi*l3eta*l2pgamma
    dershape3D[2, 24] = l1xi*l2eta*l2pgamma
    dershape3D[2, 25] = l2xi*l2eta*l3pgamma

    # center node

    shape3D[26] = l2xi*l2eta*l2gamma

    dershape3D[0, 26] = l2pxi*l2eta*l2gamma
    dershape3D[1, 26] = l2xi*l2peta*l2gamma
    dershape3D[2, 26] = l2xi*l2eta*l2pgamma

    # compute coordinates and Jacobian matrix
    x = ZERO
    y = ZERO
    z = ZERO

    xxi = ZERO
    xeta = ZERO
    xgamma = ZERO
    yxi = ZERO
    yeta = ZERO
    ygamma = ZERO
    zxi = ZERO
    zeta = ZERO
    zgamma = ZERO

    for ia in range(NGNOD):
        x = x+shape3D[ia]*xelm[ia]
        y = y+shape3D[ia]*yelm[ia]
        z = z+shape3D[ia]*zelm[ia]

        xxi = xxi+dershape3D[1, ia]*xelm[ia]
        xeta = xeta+dershape3D[2, ia]*xelm[ia]
        xgamma = xgamma+dershape3D[3, ia]*xelm[ia]
        yxi = yxi+dershape3D[1, ia]*yelm[ia]
        yeta = yeta+dershape3D[2, ia]*yelm[ia]
        ygamma = ygamma+dershape3D[3, ia]*yelm[ia]
        zxi = zxi+dershape3D[1, ia]*zelm[ia]
        zeta = zeta+dershape3D[2, ia]*zelm[ia]
        zgamma = zgamma+dershape3D[3, ia]*zelm[ia]

    jacobian = \
        xxi*(yeta*zgamma-ygamma*zeta) \
        - xeta*(yxi*zgamma-ygamma*zxi) \
        + xgamma*(yxi*zeta-yeta*zxi)

    if (jacobian <= ZERO):
        raise ValueError('3D Jacobian undefined')

    # invert the relation (Fletcher p. 50 vol. 2)
    xix = (yeta*zgamma-ygamma*zeta)/jacobian
    xiy = (xgamma*zeta-xeta*zgamma)/jacobian
    xiz = (xeta*ygamma-xgamma*yeta)/jacobian
    etax = (ygamma*zxi-yxi*zgamma)/jacobian
    etay = (xxi*zgamma-xgamma*zxi)/jacobian
    etaz = (xgamma*yxi-xxi*ygamma)/jacobian
    gammax = (yxi*zeta-yeta*zxi)/jacobian
    gammay = (xeta*zxi-xxi*zeta)/jacobian
    gammaz = (xxi*yeta-xeta*yxi)/jacobian

    return xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz
