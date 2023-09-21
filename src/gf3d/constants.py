import numpy as np

LOG_LEVEL = "WARNING"

# Some constant numbers
ONE: float = 1.0
ZERO: float = 0.0
HALF: float = 0.5
TWO: float = 2.0
TINYVAL: float = 1.0e-9
HUGEVAL: float = 1.0e30
VERYSMALLVAL: float = 1e-24

# Degrees and Radians and their conversions
PI: float = np.pi
TWO_PI: float = 2 * PI
PI_OVER_TWO: float = 0.5*PI
DEGREES_TO_RADIANS: float = PI / 180.0
RADIANS_TO_DEGREES: float = 180.0 / PI
R_UNIT_SPHERE: float = ONE
# small tolerance for conversion from x y z to r theta phi
SMALL_VAL_ANGLE = 1.0e-10

# %% This section is taken from SPECFEM3D_GLOBE's ``constants.h.in``
# For the reference ellipsoid to convert geographic latitudes to geocentric:
#
# From Dahlen and Tromp(1998): "Spherically-symmetric Earth models all have the same hydrostatic
# surface ellipticity 1/299.8. This is 0.5 percent smaller than observed flattening of best-fitting ellipsoid 1/298.3.
# The discrepancy is referred to as the "excess equatorial bulge of the Earth",
# an early discovery of artificial satellite geodesy."
#
# From Paul Melchior, IUGG General Assembly, Vienna, Austria, August 1991 Union lecture,
# available at http: // www.agu.org/books/sp/v035/SP035p0047/SP035p0047.pdf:
# "It turns out that the spheroidal models constructed on the basis of the spherically-symmetric models(PREM, 1066A)
# by using the Clairaut differential equation to calculate the flattening in function of the radius vector imply hydrostaticity.
# These have surface ellipticity 1/299.8 and a corresponding dynamical flattening of .0033 (PREM).
# The actual ellipticty of the Earth for a best-fitting ellipsoid is 1/298.3 with a corresponding dynamical flattening of .0034."
#
# Thus, flattening f = 1/299.8 is what is used in SPECFEM3D_GLOBE, as it should.
# And thus eccentricity squared e ^ 2 = 1 - (1-f) ^ 2 = 1 - (1 - 1/299.8) ^ 2 = 0.00665998813529,
# and the correction factor used in the code to convert geographic latitudes to geocentric
# is 1 - e ^ 2 = (1-f) ^ 2 = (1 - 1/299.8) ^ 2 = 0.9933400118647.
#
# As a comparison, the classical World Geodetic System reference ellipsoid WGS 84
# (see e.g. http: // en.wikipedia.org/wiki/World_Geodetic_System) has f = 1/298.2572236.
EARTH_FLATTENING_F: float = 1.0 / 299.80
EARTH_ONE_MINUS_F_SQUARED: float = (1.0 - EARTH_FLATTENING_F)**2

# EARTH_R is the radius of the bottom of the oceans(radius of Earth in m)
EARTH_R: float = 6371000.0

# and in kilometers:
EARTH_R_KM: float = EARTH_R / 1000.0

# average density in the full Earth to normalize equation
EARTH_RHOAV: float = 5514.3

# standard gravity at the surface of the Earth
EARTH_STANDARD_GRAVITY: float = 9.80665  # in m.s-2

# Even though there is ellipticity, use spherical Earth assumption for the
# conversion from geographical to spherical coordinates.
ASSUME_PERFECT_SPHERE: bool = False

# gravitational constant in S.I. units i.e. in m3 kg-1 s-2, or equivalently in N.(m/kg)^2
# DK DK April 2014: switched to the 2010 Committee on Data for Science and Technology (CODATA) recommended value
GRAV: float = 6.67384e-11  # CODATA 2010

# FOR NOW USE STATIC --> planets could be added later or specfem constants
# could be dumped.
ONE_MINUS_F_SQUARED = EARTH_ONE_MINUS_F_SQUARED
R_PLANET = EARTH_R
R_PLANET_KM = EARTH_R_KM
RHOAV = EARTH_RHOAV


# Plot colors
ORANGE = (227.0/255, 146.0/255, 60.0/255)
BLUE = (57.0/255, 137.0/255, 208.0/255)
GRAY = (183.0/255, 186.0/255, 189.0/255)
DARKGRAY = (183.0/255 * 0.75, 186.0/255 * 0.75, 189.0/255 * 0.75)
