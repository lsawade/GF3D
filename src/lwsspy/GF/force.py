"""FORCE  001

time shift:         5.0000    ! s
half duration:      3.1       ! Half duration (s) for Gaussian/Step function, frequency (Hz) for Ricker
latitude:          48.3319    ! Degree (II.BFO)
longitude:          8.3311    ! Degree
depth:              0.0000    ! km
source time function: 2       ! 0=Gaussian function, 1=Ricker wavelet, 2=Step function
factor force source: 1.d+14   ! Newton
component dir vect source E:     0.d0
component dir vect source N:     0.d0
component dir vect source Z_UP:  1.d0       ! Upward force
"""


from time import time


class Force:

    timeshift: float    # in s
    hdur: float         # in s
    latitude: float     # in deg
    longitude: float    # in deg
    depth: float        # in km
    stf: int            # 0=Gaussian function, 1=Ricker wavelet, 2=Step function
    forcefactor: float  # Newton
    vector_E: float     # East force vector
    vector_N: float     # North force vector
    vector_Z_UP: float  # Vertical UP(!) force vector
    force_no: int       # Force number

    def __init__(
        self,
        timeshift: float = 5,       # in s
        hdur: float = 3.1,          # in s
        latitude: float = 48.3319,  # in deg
        longitude: float = 8.3311,  # in deg
        depth: float = 0.0,         # in km
        stf: int = 2,               # 0=Gaussian function, 1=Ricker wavelet, 2=Step function
        forcefactor: float = 1e14,  # Newton
        vector_E: float = 0,        # East force vector
        vector_N: float = 0,        # North force vector
        vector_Z_UP: float = 1,     # Vertical UP(!) force vector
        force_no: int = 1           # Number of the force (for header)
    ) -> None:

        # Define
        self.timeshift = timeshift
        self.hdur = hdur
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.stf = stf
        self.forcefactor = forcefactor
        self.vector_N = vector_N
        self.vector_E = vector_E
        self.vector_Z_UP = vector_Z_UP
        self.force_no = force_no

    @staticmethod
    def float_to_str(x: float, N: int):

        # Check whether g formatting removes decimal points
        out = f'{x:{N}g}'

        # Fix to fortran formatting of doubles
        if '.' in out:
            if 'e' in out:
                out = out.replace('e', 'd')
            else:
                out += 'd0'
        else:
            if 'e' in out:
                out = out.replace('e', '.d')
            else:
                out += '.d0'

        return out

    def __repr__(self) -> str:

        N = f"{self.force_no:d}".zfill(3)
        rstr = f"FORCE  {N}\n"
        rstr += f"time shift:{self.timeshift:15.4f}    ! s\n"
        rstr += f"half duration:{self.hdur:9.1f}       ! Half duration (s) for Gaussian/Step function, frequency (Hz) for Ricker\n"
        rstr += f"latitude:{self.latitude:17.4f}    ! Degree \n"
        rstr += f"longitude:{self.longitude:16.4f}    ! Degree\n"
        rstr += f"depth:{self.depth:20.4f}    ! km\n"
        rstr += f"source time function:{self.stf:2d}       ! 0=Gaussian function, 1=Ricker wavelet, 2=Step function\n"
        FF = self.float_to_str(self.forcefactor, 5)
        E = self.float_to_str(self.vector_E, 3)
        N = self.float_to_str(self.vector_N, 3)
        Z = self.float_to_str(self.vector_Z_UP, 3)
        rstr += f"factor force source: {FF:<8} ! Newton\n"
        rstr += f"component dir vect source E: {E:>9}\n"
        rstr += f"component dir vect source N: {N:>9}\n"
        rstr += f"component dir vect source Z_UP: {Z:>5}  ! Upward force\n"

        return rstr

    def __str__(self):
        return self.__repr__()
