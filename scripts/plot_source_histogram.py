"""
This scripts finds the maximum hitcount of a 3D histogram based on cartesian
coordinates (not quite good for geographical coordinates)
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs

depthrange = np.array([10,15,20,30,70,120,300,700])

# minlon, maxlon, minlat maxlat
bbox = [-89.296875, -47.421875, -57.515823, 14.774883]

# Lat range
ddeg = 0.2
lonbin = np.arange(bbox[0], bbox[1], ddeg)
latbin = np.arange(bbox[2], bbox[3], ddeg)

#%%

def read_GF_LOCATIONS(file: str):
    """GF LOCATIONS must have the format """
    # Open GF locations file for each compenent
    # with open(self.target_file, 'w') as f:

    return np.loadtxt(file)


points = read_GF_LOCATIONS("workflow/GF_LOCATIONS")

# %%

H, edges = np.histogramdd(points, bins=(latbin, lonbin, depthrange))
Hn = np.where(H==0, np.nan, H)

plt.figure(figsize=(9,5))

for i in range(len(depthrange)-1):



    ax = plt.subplot(2,4,i+1, projection=ccrs.PlateCarree())
    plt.pcolormesh(lonbin, latbin, Hn[:,:,i], transform=ccrs.PlateCarree(), zorder=2)
    plt.colorbar()
    plt.title(f"{depthrange[i]} km - {depthrange[i+1]} km\nN=f{np.nansum(Hn[:,:,i])}", fontsize='small')
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='k',
                  linewidth=0.25, facecolor=(0.9, 0.9, 0.9))
    gl = ax.gridlines(lw=0.25, ls='-', color=(0.75, 0.75, 0.75), zorder=1, draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent(bbox)

plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.5, hspace=0.3)
plt.savefig('zblub.png', dpi=300)