"""
Readers and writers for FORCESOLUTION and CMTSOLUTION for specfem.


"""
import warnings
import numpy as np
from obspy import UTCDateTime
from copy import deepcopy
from obspy.imaging.beachball import beach
from matplotlib import transforms  # For beachball fix


class FORCESOLUTION:

    time_shift: float    # in s
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
        time_shift: float = 5,       # in s
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
        self.time_shift = time_shift
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

    @classmethod
    def BFO(cls, comp='Z'):

        N, E, Z = 0.0, 0.0, 0.0

        if comp == 'Z':
            Z = 1.0
        elif comp == 'E':
            E = 1.0
        elif comp == 'N':
            N = 1.0
        else:
            raise ValueError('Component must be N, E, or Z')

        return cls(
            time_shift=0.0, hdur=0.0, latitude=48.3319, longitude=8.3311,
            depth=0.0, stf=2, forcefactor=1e14, vector_E=E, vector_N=N,
            vector_Z_UP=Z, force_no=1)

    @classmethod
    def read(cls, infile: str):
        """Reads FORCESOLUTION file

        Parameters
        ----------
        infile : str
            FORCESOLUTION file

        Returns
        -------
        FORCESOLUTION class
            A class that contains all the tensor info

        """

        with open(infile, "rt") as f:
            lines = f.readlines()

        # Convert first line
        force_no = int(lines[0].strip().split()[-1])

        # Reading second line
        time_shift = float(lines[1].strip().split(':')[-1].split()[0])

        # Reading third line
        half_duration = float(lines[2].strip().split(':')[-1].split()[0])

        # Reading fourth line
        latitude = float(lines[3].strip().split(':')[-1].split()[0])

        # Reading fifth line
        longitude = float(lines[4].strip().split(':')[-1].split()[0])

        # Reading sixth line
        depth = float(lines[5].strip().split(':')[-1].split()[0])

        # Reading seventh line
        stf = int(lines[6].strip().split(':')[-1].split()[0])

        # Reading eigth line
        forcefactor = cls.str2float(lines[7].strip().split(':')[-1].split()[0])

        # Reading lines 9-11
        compE = cls.str2float(lines[8].strip().split(':')[-1].split()[0])
        compN = cls.str2float(lines[9].strip().split(':')[-1].split()[0])
        compZ_UP = cls.str2float(lines[10].strip().split(':')[-1].split()[0])

        return cls(
            time_shift=time_shift, hdur=half_duration, latitude=latitude, longitude=longitude, depth=depth, stf=stf, forcefactor=forcefactor, vector_E=compE, vector_N=compN, vector_Z_UP=compZ_UP, force_no=force_no)

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

    @staticmethod
    def str2float(x: str):

        # Fix to fortran formatting of doubles
        if "d+" in x:
            x = x.replace('d+', 'e+')
        elif "d-" in x:
            x = x.replace('d-', 'e-')
        elif "d" in x:
            x = x.replace('d', '')

        return float(x)

    def write(self, outfile: str):

        with open(outfile, 'w') as f:
            f.write(self.__repr__())

    def __repr__(self) -> str:

        N = f"{self.force_no:d}".zfill(3)
        rstr = f"FORCE  {N}\n"
        rstr += f"time shift:{self.time_shift:15.4f}    ! s\n"
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


class CMTSOLUTION:

    origin_time: UTCDateTime  # Timestamp
    pde_lat: float   # deg
    pde_lon: float   # deg
    pde_depth: float  # km
    mb: float        # magnitude scale
    ms: float        # magnitude scale
    region_tag: str  # string
    eventname: str   # event id -> GCMT
    time_shift: float  # in s
    hdur: float      # in s
    latitude: float  # in deg
    longitude: float  # in deg
    depth: float     # in m
    Mrr: float       # dyn*cm
    Mtt: float       # dyn*cm
    Mpp: float       # dyn*cm
    Mrt: float       # dyn*cm
    Mrp: float       # dyn*cm
    Mtp: float       # dyn*cm

    def __init__(
        self,
        origin_time: UTCDateTime = UTCDateTime(2000, 1, 1, 0, 0, 0),
        pde_lat: float = 0.0,     # deg
        pde_lon: float = 0.0,     # deg
        pde_depth: float = 0.0,   # km
        mb: float = 0.0,          # magnitude scale
        ms: float = 0.0,          # magnitude scale
        region_tag: str = '',     # string
        eventname: str = '',      # GCMT id
        time_shift: float = 0.0,  # in s
        hdur: float = 0.0,        # in s
        latitude: float = 0.0,    # in deg
        longitude: float = 0.0,   # in deg
        depth: float = 0.0,       # in km
        Mrr: float = 0.0,         # dyn*cm
        Mtt: float = 0.0,         # dyn*cm
        Mpp: float = 0.0,         # dyn*cm
        Mrt: float = 0.0,         # dyn*cm
        Mrp: float = 0.0,         # dyn*cm
        Mtp: float = 0.0          # dyn*cm
    ) -> None:

        # Define
        self.origin_time = origin_time
        self.pde_lat = pde_lat
        self.pde_lon = pde_lon
        self.pde_depth = pde_depth
        self.mb = mb
        self.ms = ms
        self.region_tag = region_tag
        self.eventname = eventname
        self.time_shift = time_shift
        self.hdur = hdur
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.Mrr = Mrr
        self.Mtt = Mtt
        self.Mpp = Mpp
        self.Mrt = Mrt
        self.Mrp = Mrp
        self.Mtp = Mtp

    @classmethod
    def read(cls, infile: str):
        """Reads CMT solution file

        Parameters
        ----------
        infile : str
            CMTSOLUTION file

        Returns
        -------
        CMTSOLUTION class
            A class that contains all the tensor info

        """

        with open(infile, "rt") as f:
            lines = f.readlines()

        # Convert first line
        line0 = lines[0]

        # Split up origin time values
        origin_time = line0[5:].strip().split()[:6]

        # Create datetime values
        values = list(map(int, origin_time[:-1])) + [float(origin_time[-1])]

        # Create datetime stamp
        try:
            origin_time = UTCDateTime(*values)
        except (TypeError, ValueError):
            warnings.warn("Could not determine origin time from line: %s"
                          % line0)
            origin_time = UTCDateTime(0)

        otherinfo = line0[4:].strip().split()[6:]
        pde_lat = float(otherinfo[0])
        pde_lon = float(otherinfo[1])
        pde_depth = float(otherinfo[2])
        mb = float(otherinfo[3])
        ms = float(otherinfo[4])
        region_tag = ' '.join(otherinfo[5:])

        # Reading second line
        eventname = lines[1].strip().split()[-1]

        # Reading third line
        time_shift = float(lines[2].strip().split()[-1])

        # Reading fourth line
        half_duration = float(lines[3].strip().split()[-1])

        # Reading fifth line
        latitude = float(lines[4].strip().split()[-1])

        # Reading sixth line
        longitude = float(lines[5].strip().split()[-1])

        # Reading seventh line
        depth = float(lines[6].strip().split()[-1])

        # Reading lines 8-13
        Mrr = float(lines[7].strip().split()[-1])
        Mtt = float(lines[8].strip().split()[-1])
        Mpp = float(lines[9].strip().split()[-1])
        Mrt = float(lines[10].strip().split()[-1])
        Mrp = float(lines[11].strip().split()[-1])
        Mtp = float(lines[12].strip().split()[-1])

        return cls(
            origin_time=origin_time,
            pde_lat=pde_lat,
            pde_lon=pde_lon,
            pde_depth=pde_depth,
            mb=mb,
            ms=ms,
            region_tag=region_tag,
            eventname=eventname,
            time_shift=time_shift,
            hdur=half_duration,
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            Mrr=Mrr,
            Mtt=Mtt,
            Mpp=Mpp,
            Mrt=Mrt,
            Mrp=Mrp,
            Mtp=Mtp)

    @property
    def tensor(self):
        return np.array([self.Mrr, self.Mtt, self.Mpp, self.Mrt, self.Mrp, self.Mtp])

    @property
    def cmt_time(self):
        return self.origin_time + self.time_shift

    @property
    def M0(self):
        """
        Scalar Moment M0 in Nm
        """
        return (self.Mrr ** 2 + self.Mtt ** 2 + self.Mpp ** 2
                + 2 * self.Mrt ** 2 + 2 * self.Mrp ** 2
                + 2 * self.Mtp ** 2) ** 0.5 * 0.5 ** 0.5

    @M0.setter
    def M0(self, M0):
        """
        Scalar Moment M0 in Nm
        """
        iM0 = self.M0
        fM0 = M0
        factor = fM0/iM0
        self.Mrr *= factor
        self.Mtt *= factor
        self.Mpp *= factor
        self.Mrt *= factor
        self.Mrp *= factor
        self.Mtp *= factor

        self.update_hdur()

    @property
    def Mw(self):
        """
        Moment magnitude M_w
        """
        return 2/3 * np.log10(7 + self.M0) - 10.73

    def pert(self, param: str, pert: float):

        outcmt = deepcopy(self)

        setattr(outcmt, param, getattr(outcmt, param) + pert)

        return outcmt

    def update_hdur(self):
        # Updates the half duration
        Nm_conv = 1 / 1e7
        self.half_duration = np.round(
            2.26 * 10**(-6) * (self.M0 * Nm_conv)**(1/3), decimals=1)

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

    def axbeach(
            self, ax, x, y, width=50, facecolor='k', linewidth=2, alpha=1.0,
            clip_on=False, **kwargs):
        """Plots beach ball into given axes.
        Note that width heavily depends on the given screen size/dpi. Therefore
        often does not work."""

        # Plot beach ball
        bb = beach(self.tensor,
                   linewidth=linewidth,
                   facecolor=facecolor,
                   bgcolor='w',
                   edgecolor='k',
                   alpha=alpha,
                   xy=(x, y),
                   width=width,
                   size=100,  # Defines number of interpolation points
                   axes=ax,
                   **kwargs)
        bb.set(clip_on=clip_on)

        # This fixes pdf output issue
        bb.set_transform(transforms.Affine2D(np.identity(3)))

        ax.add_collection(bb)

    def write(self, outfile: str):

        with open(outfile, 'w') as f:
            f.write(self.__str__())

    def __str__(self):

        # Reconstruct the first line as well as possible. All
        # hypocentral information is missing.
        if isinstance(self.origin_time, UTCDateTime):
            return_str = \
                " PDE %4i %2i %2i %2i %2i %5.2f %8.4f %9.4f %5.1f %.1f %.1f" \
                " %s\n" % (
                    self.origin_time.year,
                    self.origin_time.month,
                    self.origin_time.day,
                    self.origin_time.hour,
                    self.origin_time.minute,
                    self.origin_time.second
                    + self.origin_time.microsecond / 1E6,
                    self.pde_lat,
                    self.pde_lon,
                    self.pde_depth,
                    self.mb,
                    self.ms,
                    self.region_tag)
        else:
            return_str = "----- CMT Delta: ------- \n"

        return_str += 'event name:  %10s\n' % (str(self.eventname),)
        return_str += 'time shift:%12.4f\n' % (self.time_shift,)
        return_str += 'half duration:%9.4f\n' % (self.hdur,)
        return_str += 'latitude:%14.4f\n' % (self.latitude,)
        return_str += 'longitude:%13.4f\n' % (self.longitude,)
        return_str += 'depth:%17.4f\n' % (self.depth,)
        return_str += 'Mrr:%19.6e\n' % (self.Mrr,)
        return_str += 'Mtt:%19.6e\n' % (self.Mtt,)
        return_str += 'Mpp:%19.6e\n' % (self.Mpp,)
        return_str += 'Mrt:%19.6e\n' % (self.Mrt,)
        return_str += 'Mrp:%19.6e\n' % (self.Mrp,)
        return_str += 'Mtp:%19.6e\n' % (self.Mtp,)

        return return_str

    def __repr__(self) -> str:
        return self.__str__()
