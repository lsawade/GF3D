from lwsspy.GF.simulation import Simulation
from lwsspy.GF.utils import read_toml
# from lwsspy.seismo.
from copy import deepcopy
from nnodes import Node
from nnodes.job import Slurm
import toml
import os


# These need to be loaded from a STATIONS file
# network = "II"
# station = "BFO"
# station_latitude = 48.3319
# station_longitude = 8.3311
# station_burial = 0.000


# Filenames in DB
# db = '/scratch/gpfs/lsawade/SpecfemMagicGF/DB'
# stationdir = os.path.join(db, network, station)
# compdict = dict()
# for _i, comp in enumerate(['N', 'E', 'Z']):
#     compdict[comp] = os.path.join(
#         stationdir, comp, 'specfem')


def main(node: Node):
    """
    """

    # Read station list

    # Print welcome message using parameter from config.toml
    print(f"{node.workflowname}")

    # Set node to concurrent
    node.concurrent = True
    networks, stations, latitudes, longitudes, _, burials, _ = read_stations(
        node.station_file)

    # Add station configuration
    for net, sta, lat, lon, bur in zip(networks, stations, latitudes, longitudes, burials):

        # Add workflow for a single station
        node.add(station, concurrent=False, name=f'{net}.{sta}',
                 network=net,
                 station=sta,
                 latitude=lat,
                 longitude=lon,
                 burial=bur/1000.0,
                 stationdir=os.path.join(node.db, net, sta))  # burial is in meters so divide by 1000.0


def station(node: Node):

    print(node.network, node.station, node.latitude, node.longitude, node.burial)

    # Create station directory in the database.
    if node.creation:
        node.add(create_station_dir,
                 name=f'Create-Dir.-for-{node.network}.{node.station}')

    # Simulation
    if node.simulation:
        node.add(
            simulation, name=f'Simulations-for-{node.network}.{node.station}')

    # Processing
    if node.processing:
        node.add(
            processing, name=f'Processing-for-{node.network}.{node.station}')

    if node.clear:
        node.add(clear, name=f'Clearing-for-{node.network}.{node.station}')


def create_station_dir(node: Node):

    config = deepcopy(node.cfg)
    config['stationdir'] = os.path.join(node.db, node.network, node.station)
    config['network'] = node.network
    config['station'] = node.station
    config['station_burial'] = node.burial
    config['station_latitude'] = node.latitude
    config['station_longitude'] = node.longitude
    config['target_file'] = os.path.abspath(config['target_file'])

    # Setup
    S = Simulation(**config)
    print(S)
    S.create(no_specfem=True)

    # dump config
    with open(os.path.join(node.db, node.network, node.station, 'config.toml'), 'w') as f:
        toml.dump(config, f)


def simulation(node: Node):

    # The way that traverse should run these is gpus-per-proc*mps is the number
    # of GPUs requested for the mps server
    node.concurrent = True

    for comp in ['N', 'E', 'Z']:
        compsimdir = os.path.join(node.stationdir, comp, 'specfem')
        node.add_mpi('./bin/xspecfem3D', nprocs=24, gpus_per_proc=4, mps=3,
                     cwd=compsimdir,
                     name=f'sim-{node.network}.{node.station}.{comp}',
                     exec_args={Slurm: '-N2 --exclusive'})


def processing(node: Node):

    # Args for the mpi script
    # h5file, Nfile, Efile, Zfile, config_file, precision, compression
    h5file = os.path.join(node.stationdir, f'{node.network}.{node.station}.h5')
    # DB/II/BFO/N/specfem/OUTPUT_FILES/save_forward_arrays_GF.bp
    filedict = dict()
    for comp in ['N', 'E', 'Z']:
        filedict[comp] = os.path.join(node.stationdir, comp, 'specfem',
                                      'OUTPUT_FILES', 'save_forward_arrays_GF.bp')

    config_file = os.path.join(node.stationdir, 'config.toml')
    precision = node.processparams['precision']
    compression = node.processparams['compression']

    # Getting the command script
    cmd = str(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'scripts', 'processadios.py'
    ))
    args = f"{h5file} {filedict['N']} {filedict['E']} " \
           f"{filedict['Z']} {config_file} {precision} {compression}"

    # Combine all args
    cmd = f"python {cmd} {args}"

    print(node.stationdir)
    print(cmd)

    node.add_mpi(cmd, nprocs=3,
                 cwd=node.stationdir,
                 name=f'Processing-{node.network}-{node.station}',
                 priority=1, exec_args={Slurm: '-N1 --mem=199GB'})


def clear(node: Node):

    for comp in ['N', 'E', 'Z']:
        sdir = os.path.join(node.stationdir, comp, 'specfem')

        if os.path.exists(sdir):
            node.add_mpi(
                f'rm -rf {sdir}', nprocs=1, priority=2,
                name=f'process-{node.network}.{node.station}',
                exec_args={Slurm: '-N1'})


def read_stations(stations_file):

    net = []
    sta = []
    lat = []
    lon = []
    ele = []
    bur = []
    sen = []

    # station file has following format:
    # 'Net.Sta', 'Lat', 'Lon', 'Elev', 'Bur', Sensor
    with open(stations_file, 'r') as f:

        for line in f.readlines():
            line = line.strip()
            if line[0] == '#':
                continue
            line = line.split(',')
            _net, _sta = line[0].split('.')
            net.append(_net.strip())
            sta.append(_sta.strip())
            lat.append(float(line[1]))
            lon.append(float(line[2]))
            ele.append(float(line[3]))
            bur.append(float(line[4]))
            sen.append(line[5])

    return net, sta, lat, lon, ele, bur, sen
