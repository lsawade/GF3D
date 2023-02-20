from obspy.clients.fdsn import Client
from obspy.clients.fdsn.mass_downloader import RectangularDomain, \
    Restrictions, MassDownloader
from obspy import UTCDateTime
from obspy import Inventory
from typing import Union, List
import os
import logging
from typing import Union, Tuple
from obspy import UTCDateTime, Stream, Inventory
from obspy.clients.fdsn import Client


def download_stream(origintime: UTCDateTime, duration: float = 7200,
                    network: Union[str, None] = "IU,II",
                    station: Union[str, None] = None,
                    location: Union[str, None] = "00",
                    channel: Union[str, None] = "BH*",
                    starttimeoffset: float = 0.0,
                    endtimeoffset: float = 0.0, dtype='both',
                    client_id: str = "IRIS",
                    ) -> Tuple[Stream, Inventory]:
    """Function to download data for a seismic section. Note that this will not
    store the data. It will only download the data into a Stream object, and
    optionally into a corresponding inventory.

    Parameters
    ----------
    origintime : UTCDateTime
        origintime of an earthquake
    duration : float, optional
        length of download in seconds, by default 7200
    network : str or None, optional
        Network restrictions, by default "IU,II"
    station : str or None, optional
        station restrictions, by default None
    location : str or None, optional
        location restrictions, by default "00"
    channel : str or None, optional
        channel restrictions, by default "BH*"
    starttimeoffset : float, optional
        set startime to later or earlier, by default 0.0
    endtimeoffset : float, optional
        set endtime to earlier or later, by default 0.0

    Returns
    -------
    Tuple[Stream, Inventory]
        tuple with a stream and an inventory

    Raises
    ------

    ValueError
        If wrong download type is provided.

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.01.13 11.00

    """

    if dtype not in ['data', 'stations', 'both']:
        raise ValueError(
            "download type must be 'data', 'stations', or 'both'.")

    # Get times
    starttime = origintime + starttimeoffset
    endtime = origintime + duration + endtimeoffset

    # main program
    client = Client(client_id)

    # Download the data
    if (dtype == 'both') or (dtype == "data"):
        st = client.get_waveforms(network, station, location, channel,
                                  starttime, endtime)
    if (dtype == 'both') or (dtype == "stations"):
        inv = client.get_stations(network=network, station=station,
                                  location=location, channel=channel,
                                  starttime=starttime, endtime=endtime,
                                  level="response")
    if dtype == 'both':
        return st, inv
    elif dtype == 'stations':
        return inv
    elif dtype == 'data':
        return st


def download_to_storage(
        datastorage: str,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        minimum_length: float = 0.9,
        reject_channels_with_gaps: bool = True,
        network: Union[str, None] = "IU,II,G",
        station: Union[str, None] = None,
        channel: Union[str, None] = None,
        location: Union[str, None] = None,
        providers: Union[List[str], None] = ["IRIS"],
        minlatitude: float = -90.0,
        maxlatitude: float = 90.0,
        minlongitude: float = -180.0,
        maxlongitude: float = 180.0,
        location_priorities=None,
        channel_priorities=None,
        limit_stations_to_inventory: Union[Inventory, None] = None,
        waveform_storage: str = None,
        station_storage: str = None,
        logfile: str = None,
        client: Client | List[Client] | None = None,
        **kwargs):

    domain = RectangularDomain(minlatitude=minlatitude,
                               maxlatitude=maxlatitude,
                               minlongitude=minlongitude,
                               maxlongitude=maxlongitude)

    # Create Dictionary with the settings
    rdict = dict(
        starttime=starttime,
        endtime=endtime,
        reject_channels_with_gaps=reject_channels_with_gaps,
        # Trace needs to be almost full length
        minimum_length=minimum_length,
        network=network,
        station=station,
        location=location,
        channel=channel,
        location_priorities=location_priorities,
        channel_priorities=channel_priorities,
        limit_stations_to_inventory=limit_stations_to_inventory,
        sanitize=True
    )

    # Remove unset settings
    if not location_priorities:
        rdict.pop('location_priorities')
    if not channel_priorities:
        rdict.pop('channel_priorities')

    restrictions = Restrictions(**rdict)

    # Datastorage:
    if waveform_storage is None:
        waveform_storage = os.path.join(datastorage, 'waveforms')
    if station_storage is None:
        station_storage = os.path.join(datastorage, 'stations')

    # Get the logger from the obspy package
    logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")

    # Setup the logger to print to file instead of stdout/-err
    if logfile is not None:
        # Remove Stream handler (prints stuff to stdout)
        logger.handlers = []

        # Add File handler (prints stuff to file)
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)

        # Add file handler
        logger.addHandler(fh)

    if client is not None:
        providers = client

    # Create massdownloader
    mdl = MassDownloader(providers=providers)
    logger.debug(f"\n")
    logger.debug(f"{' Downloading data to: ':*^72}")
    logger.debug(f"MSEEDs: {waveform_storage}")
    logger.debug(f"XMLs:   {station_storage}")

    mdl.download(domain, restrictions, mseed_storage=waveform_storage,
                 stationxml_storage=station_storage, **kwargs)

    logger.debug("\n")
    logger.debug(72 * "*")
    logger.debug("\n")
