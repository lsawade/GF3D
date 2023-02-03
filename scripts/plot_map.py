# %%
import os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from lwsspy.seismo import CMTCatalog
from lwsspy.GF.utils import filedir


DIRNAME = f'{os.getenv("HOME")}/lwsspy/lwsspy.GF/scripts'
# %%

cat = CMTCatalog.load(os.path.join(DIRNAME, 'DATA/gcmt_catalog_20220924.pkl'))

# %%
# South America Bbox
bbox = [-89.296875, -47.421875, -57.515823, 14.774883]

# %% Define minmax dictionaries from bounding box
mindict = dict(latitude=bbox[2], longitude=bbox[0])
maxdict = dict(latitude=bbox[3], longitude=bbox[1])

# %%
cat, _ = cat.filter(mindict=mindict, maxdict=maxdict)

# %%


plt.figure(figsize=(5, 10))

mapax = plt.axes(projection=ccrs.Orthographic(
    central_longitude=sum(bbox[:2])/2, central_latitude=sum(bbox[2:])/2))
mapax.set_extent(bbox)

mapax.spines['geo'].set_visible(False)

cat.plot(ax=mapax)

plt.savefig('eventmap.pdf', dpi=300)
