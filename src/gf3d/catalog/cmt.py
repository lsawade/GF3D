import os
import numpy as np
import pickle
from ..source import CMTSOLUTION
from typing import List, Iterable
from ..source import CMTSOLUTION
from .utils import cmts2dir, cmts2file
from .download import download_gcmt_catalog
from obspy import read_events, Catalog as obspycat

class CMTCatalog(List[CMTSOLUTION]):

    def __init__(self, cmts: Iterable[CMTSOLUTION] | obspycat):

        # Make actual cmts from obspy catalog
        if isinstance(cmts, obspycat):
            cmts = [CMTSOLUTION.from_event(event) for event in cmts]

        # Return CMTcatalog
        super().__init__(cmts)

    def filter(self, param: str, low=-np.inf, high=np.inf):
        """ Filter to filter out entire catalog.

        Parameters
        ----------
        param : str
            parameter to filter
        low : _type_, optional
            low cut value. Everything below is removed, by default -np.inf
        high : _type_, optional
            high cut value. Everything above is removed, by default np.inf

        Returns
        -------
        _type_
            _description_
        """

        #First get a copy of the catalog
        ncmts, delcmts = [],[]
        # Get the relevant property
        prop = self.__getattribute__(param)

        # prop length
        N = len(self)

        # Check which indeces are being kept.
        idx = (low < prop) & (prop < high)

        for _i, _bool in enumerate(idx):
            if _bool:
                ncmts.append(self[_i])
            else:
                delcmts.append(self[_i])

        return CMTCatalog(ncmts), CMTCatalog(delcmts)

    def cmts2dir(self,  outdir: str = "./catalog"):
        """Writes catalog as seperate cmt files to directory.
        See: :func:`gf3d.catalog.utils.cmts2dir`."""
        cmts2dir(self,  outdir=outdir)

    def cmts2file(self, outfile: str = './catalog.txt'):
        """Writes catalog to a single file for, e.g., finite-fault simulations.
        See: :func:`gf3d.catalog.utils.cmts2file`."""
        cmts2file(self, outfile=outfile)

    def save(self, filename: str):
        """Store catalog as pickle file

        Parameters
        ----------
        filename : str
            filename to store catalog under
        """

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_gcmt(cls, gcmtndkfilename: str):

        # First
        if not os.path.exists(gcmtndkfilename):
            download_gcmt_catalog(gcmtndkfilename)

        # load obspy catalog
        cat = read_events(gcmtndkfilename)

        return cls(cat)


    @classmethod
    def load(cls, filename: str):
        """Load CMTCatalog that was stored as a pickle.

        Parameters
        ----------
        filename : str
            filename to load

        Returns
        -------
        CMTCatalog

        """
        with open(filename, 'rb') as f:
            cls = pickle.load(f)
        return cls

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):

        Mw = self.Mw
        utc = self.origin_time
        lat = self.latitude
        lon = self.longitude

        string = "CMTCatalog :\n"
        string += 60 * "-" + "\n"
        string += "\n"
        string += f"Event #:{len(self):.>52}\n"
        string += f"Starttime:{min(utc).strftime('%Y/%m/%d, %H:%M:%S'):.>50}\n"
        string += f"Endtime:{max(utc).strftime('%Y/%m/%d, %H:%M:%S'):.>52}\n"
        string += f"Bounding Box:{'Latitude: ':.>25}{'':.<2}[{np.min(lat):8.3f}, {np.max(lat):8.3f}]\n"
        string += f"{'Longitude: ':.>39}{'':.<1}[{np.min(lon):8.3f}, {np.max(lon):8.3f}]\n"
        string += f"Moment Magnitude:{'':.>23}[{np.min(Mw):8.3f}, {np.max(Mw):8.3f}]\n"
        string += "\n"
        string += 60 * "-" + "\n"

        return string

    @property
    def latitude(self):
        return np.array([cmt.latitude for cmt in self])

    @property
    def longitude(self):
        return np.array([cmt.longitude for cmt in self])

    @property
    def depth(self):
        return np.array([cmt.depth for cmt in self])

    @property
    def cmt_time(self):
        return np.array([cmt.cmt_time for cmt in self])

    @property
    def origin_time(self):
        return np.array([cmt.origin_time for cmt in self])

    @property
    def time_shift(self):
        return np.array([cmt.time_shift for cmt in self])

    @property
    def hdur(self):
        return np.array([cmt.hdur for cmt in self])

    @property
    def M0(self):
        return np.array([cmt.M0 for cmt in self])

    @property
    def Mw(self):
        return np.array([cmt.Mw for cmt in self])

    @property
    def eventname(self):
        return [cmt.eventname for cmt in self]

    @property
    def pde_lat(self):
        return np.array([cmt.pde_lat for cmt in self])

    @property
    def pde_lon(self):
        return np.array([cmt.pde_lon for cmt in self])

    @property
    def pde_depth(self):
        return np.array([cmt.pde_depth for cmt in self])

    @property
    def mb(self):
        return np.array([cmt.mb for cmt in self])

    @property
    def ms(self):
        return np.array([cmt.ms for cmt in self])

    @property
    def Mrr(self):
        return np.array([cmt.Mrr for cmt in self])

    @property
    def Mtt(self):
        return np.array([cmt.Mtt for cmt in self])

    @property
    def Mpp(self):
        return np.array([cmt.Mpp for cmt in self])

    @property
    def Mrt(self):
        return np.array([cmt.Mrt for cmt in self])

    @property
    def Mrp(self):
        return np.array([cmt.Mrp for cmt in self])

    @property
    def Mtp(self):
        return np.array([cmt.Mtp for cmt in self])

    @property
    def tensor(self):
        return np.vstack([cmt.tensor for cmt in self])

    def plot_global_map(self, outfile='global_eventmap.pdf', *args, **kwargs):
        """Plot global map of events in catalog.

        Parameters
        ----------
        outfile : str, optional
            filename to save figure to, by default 'global_eventmap.pdf'
        """

        from .plot import plot_global_gmt_map

        plot_global_gmt_map(self, outfile=outfile, *args, **kwargs)















