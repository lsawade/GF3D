
from urllib.error import URLError
from .utils import downloadfile, get_url_content
import ast
import functools
try:
    from tqdm import tqdm
    from .utils import downloadfile_progress as downloadfile
except ImportError:
    from .utils import downloadfile

urlerrorans = \
    """
If the Error below is
    "<urlopen error [Errno 61] Connection refused>"
you probably don't have access to the database. Make sure you setup the ssh
tunnel correctly if you do.

ssh -f -N -F 'none' -L 5000:127.0.0.1:5000 <username>@vrientius.princeton.edu

If the error below shows:
    "[Errno 54] Connection reset by peer"
you probably have to re-login to your VPN, or check you multiplexed connection

Here's the error in case it's not that:
"""


def catch_exceptions(job_func):
    @functools.wraps(job_func)
    def wrapper(*args, **kwargs):
        try:
            return job_func(*args, **kwargs)
        except Exception as e:
            print(urlerrorans)
            print(e)
            return None
    return wrapper


class GF3DClient:

    # Database name
    db: str

    # Server info
    base_url: str = '127.0.0.1'
    port: int = 5000

    # Server pages
    info_route: str = 'get-db-info'
    avail_route: str = 'get-station-availability'
    subset_route: str = 'get-subset'

    def __init__(self,
                 db: str = 'example-db',
                 debug: bool = False):
        """Initializes a client class that you can get a

        Parameters
        ----------
        db : str, optional
            database , by default 'example-db'
        """

        self.db = db
        self.debug = debug

    @catch_exceptions
    def stations_avail(self):
        url = f'http://{self.base_url}:{self.port:d}/{self.avail_route}?'
        url += f"db={self.db}"

        if self.debug:
            print(url)
        else:
            return get_url_content(url).decode().split(',')

    @catch_exceptions
    def get_info(self):
        url = f'http://{self.base_url}:{self.port:d}/{self.info_route}?'
        url += f"db={self.db}"

        if self.debug:
            print(url)
        else:
            result = get_url_content(url).decode()
            return ast.literal_eval(result)

    @catch_exceptions
    def get_subset(self, outputfile: str,
                   latitude: float, longitude: float, depth_in_km: float, radius_in_km: float = 100,
                   NGLL: int = 5, netsta: list | None = None, fortran: bool = False):
        """Download a subset of stations from a database server. The example
        database has a single element with the following coordinates:
        (latitude=-31.1300, longitude=-72.0900, depth=17.3500)

        Parameters
        ----------
        lat : float
            latitude of request center
        lon : float
            longitude of request center
        depth_in_km : float
            depth of request center in km
        radius_in_km : float, optional
            radius, by default 100
        netsta : list | None, optional
            List of stations to request, by default None
        fortran : bool, optional
            whether to return the HDF5 file with fortran array ordering,
            by default False
        """

        url = f'http://{self.base_url}:{self.port}/{self.subset_route}?'
        url += f"db={self.db}"
        url += f"&latitude={latitude:f}"
        url += f"&longitude={longitude:f}"
        url += f"&depth={depth_in_km:f}"
        url += f"&radius={radius_in_km:f}"
        url += f"&NGLL={NGLL:d}"

        # Check whether we can get selected stations only
        if netsta is not None:

            # Check formatting of station
            if isinstance(netsta[0], str) is False:
                raise ValueError('Check your netsta setup')

            # Check formatting of station 2
            if netsta[0].split('.') != 2:
                raise ValueError('Check your netsta setup')

            # Check against available stations
            available_stations = list(self.stations_avail())
            for _netsta in netsta:
                if _netsta not in available_stations:
                    raise ValueError(
                        f'{_netsta} not in database. available stations: '
                        f'{available_stations}')

            stationstr = '[' + \
                ','.join([f'"{_netsta}"' for _netsta in netsta]) + ']'
            url += f"&netsta={stationstr}"

        if isinstance(fortran, bool) and fortran:
            url += f"&fortran={fortran}"

        if self.debug:
            print(url)
        else:
            downloadfile(url, outputfile, desc='Downloading created subset')
