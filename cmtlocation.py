# %% import_stmt
from lwsspy.GF.source2xyz import source2xyz
from lwsspy.GF.source import CMTSOLUTION

# %% Get CMT solution to convert
cmt = CMTSOLUTION.read('CMTSOLUTION')


cmt: CMTSOLUTION,
topography = True,
ellipticity = True
ibathy_topo: np.ndarray | None = None,



    source2xyz(
        cmt.latitude,
        cmt.longitude,
        cmt.depth_in_km,
        cmt.tensor,
        topography=topography,
        ellipticity=ellipticity,
        ibathy_topo: np.ndarray | None = None,
        NX_BATHY: int | None = None,
        NY_BATHY: int | None = None,
        RESOLUTION_TOPO_FILE: float | None = None,
        rspl: np.ndarray | None = None,
        ellipicity_spline: np.ndarray | None = None,
        ellipicity_spline2: np.ndarray | None = None
    )
