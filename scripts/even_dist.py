# %%
from obspy.geodetics.base import locations2degrees
import numpy as np
import matplotlib.pyplot as plt

# %%
sta_list = [
    ['IC', 'WMQ', 43.8138, 87.7049, 0.006],
    ['IC', 'HIA', 49.2704, 119.7414, 0.0],
    ['IC', 'XAN', 34.0313, 108.9237, 0.0],
    ['IC', 'MDJ', 44.617, 129.5908, 0.05],
    ['IC', 'BJT', 40.0183, 116.1679, 0.06],
    ['IC', 'QIZ', 19.0291, 109.8445, 0.0],
    ['IC', 'ENH', 30.2762, 109.4944, 0.0],
    ['IC', 'LSA', 29.7031, 91.127, 0.015],
    ['IC', 'KMI', 25.1233, 102.74, 0.035],
    ['IC', 'SSE', 31.0948, 121.1908, 0.0],
    ['II', 'EFI', -51.6753, -58.0637, 0.08],
    ['II', 'KIV', 43.9562, 42.6888, 0.0],
    ['II', 'ABPO', -19.018, 47.229, 0.0053],
    ['II', 'ABKT', 37.9304, 58.1189, 0.007],
    ['II', 'NVS', 54.8404, 83.2346, 0.0],
    ['II', 'HOPE', -54.2836, -36.4879, 0.0],
    ['II', 'KWJN', 9.2873, 167.5369, 0.006],
    ['II', 'ALE', 82.5033, -62.35, 0.0],
    ['II', 'MSVF', -17.7448, 178.0528, 0.1],
    ['II', 'PFO', 33.6092, -116.4553, 0.0],
    ['II', 'ASCN', -7.9327, -14.3601, 0.1],
    ['II', 'ERM', 42.015, 143.1572, 0.0],
    ['II', 'TLY', 51.6807, 103.6438, 0.02],
    ['II', 'JTS', 10.2908, -84.9525, 0.0],
    ['II', 'PALK', 7.2728, 80.7022, 0.09],
    ['II', 'COCO', -12.1901, 96.8349, 0.07],
    ['II', 'BRVK', 53.0581, 70.2828, 0.015],
    ['II', 'WRAB', -19.9336, 134.36, 0.1],
    ['II', 'SIMI', 38.6585, 69.0083, 0.0],
    ['II', 'ARU', 56.4302, 58.5625, 0.0],
    ['II', 'GAR', 39.0052, 70.3328, 0.0],
    ['II', 'ARTI', 56.3879, 58.3849, 0.0045],
    ['II', 'OBN', 55.1146, 36.5674, 0.03],
    ['II', 'NRIL', 69.5049, 88.4414, 0.506],
    ['II', 'FFC', 54.725, -101.9783, 0.0],
    ['II', 'MBAR', -0.6019, 30.7382, 0.1],
    ['II', 'KDAK', 57.7828, -152.5835, 0.088],
    ['II', 'XPFO', 33.6107, -116.4555, 0.0],
    ['II', 'KWAJ', 8.8019, 167.613, 0.0],
    ['II', 'NNA', -11.9875, -76.8422, 0.04],
    ['II', 'SACV', 14.9702, -23.6085, 0.097],
    ['II', 'RAYN', 23.5225, 45.5032, 0.075],
    ['II', 'CMLA', 37.7637, -25.5243, 0.097],
    ['II', 'TAU', -42.9099, 147.3204, 0.0],
    ['II', 'KURK', 50.7154, 78.6202, 0.025],
    ['II', 'BFO', 48.3301, 8.3296, 0.16],
    ['II', 'ESK', 55.3167, -3.205, 0.0],
    ['II', 'SUR', -32.3797, 20.8117, 0.0],
    ['II', 'NIL', 33.6506, 73.2686, 0.068],
    ['II', 'DGAR', -7.4121, 72.4525, 0.002],
    ['II', 'BORK', 53.0461, 70.3184, 0.0425],
    ['II', 'AAK', 42.6375, 74.4942, 0.03],
    ['II', 'XPF', 33.6092, -116.4533, 0.1],
    ['II', 'RPN', -27.1267, -109.3344, 0.0],
    ['II', 'MSEY', -4.6737, 55.4792, 0.091],
    ['II', 'LVZ', 67.8979, 34.6514, 0.2],
    ['II', 'UOSS', 24.9453, 56.2042, 0.0],
    ['II', 'KAPI', -5.0142, 119.7517, 0.1],
    ['II', 'SHEL', -15.9594, -5.7455, 0.06],
    ['II', 'BORG', 64.7474, -21.3268, 0.095],
    ['G', 'MPG', 5.1101, -52.6445, 0.0],
    ['G', 'ROCAM', -19.7555, 63.3701, 0.0],
    ['G', 'CAN', -35.3187, 148.9963, 0.0],
    ['G', 'EDA', 3.7789, 10.1534, 0.0],
    ['G', 'ATD', 11.5307, 42.8466, 0.0],
    ['G', 'FDF', 14.735, -61.1463, 0.0],
    ['G', 'TAM', 22.7915, 5.5284, 0.0],
    ['G', 'INU', 35.35, 137.029, 0.0],
    ['G', 'FDFM', 14.7349, -61.1597, 0.0094],
    ['G', 'SSB', 45.279, 4.542, 0.0],
    ['G', 'PAF', -49.351, 70.2107, 0.0],
    ['G', 'GRC', 47.2955, 3.0736, 0.0],
    ['G', 'DRV', -66.6649, 140.0021, 0.004],
    ['G', 'RODM', -19.6962, 63.4413, 0.0],
    ['G', 'UNM', 19.3297, -99.1781, 0.0],
    ['G', 'HDC', 10.0021, -84.1114, 0.0],
    ['G', 'FUTU', -14.3077, -178.1211, 0.0],
    ['G', 'CRZF', -46.43, 51.861, 0.002],
    ['G', 'SANVU', -15.4471, 167.2032, 0.0],
    ['G', 'AIS', -37.7963, 77.5692, 0.003],
    ['G', 'NOUC', -22.0986, 166.3066, 0.0],
    ['G', 'CLF', 48.0258, 2.26, 0.0],
    ['G', 'CCD', -75.1065, 123.305, 0.0],
    ['G', 'SPB', -23.5927, -47.427, 0.0],
    ['G', 'COYC', -45.573, -72.0814, 0.0],
    ['G', 'IVI', 61.2058, -48.1712, 0.0],
    ['G', 'SOK', 14.4944, -16.4562, 0.004200000000000001],
    ['G', 'KIP', 21.42, -158.0112, 0.033],
    ['G', 'WUS', 41.2007, 79.2165, 0.016],
    ['G', 'TRIS', -37.0681, -12.3152, 0.002],
    ['G', 'PEL', -33.1436, -70.6749, 0.0],
    ['G', 'FOMA', -24.9757, 46.9789, 0.0],
    ['G', 'RER', -21.1712, 55.7399, 0.0],
    ['G', 'ECH', 48.2163, 7.159, 0.25],
    ['G', 'PPTF', -17.5896, -149.5652, 0.0],
    ['G', 'MBO', 14.392, -16.9555, 0.002],
    ['IU', 'JOHN', 16.7329, -169.5292, 0.039],
    ['IU', 'TRQA', -38.0568, -61.9787, 0.101],
    ['IU', 'PMG', -9.4047, 147.1597, 0.0],
    ['IU', 'COLA', 64.8736, -147.8616, 0.12],
    ['IU', 'HKT', 29.9618, -95.8384, 0.45],
    ['IU', 'NWAO', -32.9277, 117.239, 0.105],
    ['IU', 'HRV', 42.5064, -71.5583, 0.0],
    ['IU', 'CTAO', -20.0882, 146.2545, 0.037],
    ['IU', 'SFJD', 66.9961, -50.6208, 0.0],
    ['IU', 'YSS', 46.9587, 142.7604, 0.002],
    ['IU', 'TSUM', -19.2022, 17.5838, 0.0],
    ['IU', 'SNZO', -41.3087, 174.7043, 0.1],
    ['IU', 'YAK', 62.031, 129.6805, 0.014],
    ['IU', 'PAB', 39.5446, -4.3499, 0.0],
    ['IU', 'ULN', 47.8651, 107.0532, 0.0],
    ['IU', 'RCBR', -5.8274, -35.9014, 0.109],
    ['IU', 'MBWA', -21.159, 119.7313, 0.102],
    ['IU', 'SAML', -8.9489, -63.1831, 0.111],
    ['IU', 'LVC', -22.6127, -68.9111, 0.03],
    ['IU', 'FURI', 8.8952, 38.6798, 0.005],
    ['IU', 'CCM', 38.0557, -91.2446, 0.051],
    ['IU', 'KMBO', -1.1271, 37.2525, 0.02],
    ['IU', 'SBA', -77.8492, 166.7572, 0.002],
    ['IU', 'MAJO', 36.5457, 138.2041, 0.0],
    ['IU', 'ANMO', 34.9459, -106.4572, 0.15],
    ['IU', 'TATO', 24.9735, 121.4971, 0.0835],
    ['IU', 'INCN', 37.4776, 126.6239, 0.015],
    ['IU', 'KEV', 69.7565, 27.0035, 0.015],
    ['IU', 'PAYG', -0.6742, -90.2861, 0.1],
    ['IU', 'SPA', -90.0, 0.0, 0.01],
    ['IU', 'MSKU', -1.6557, 13.6116, 0.025],
    ['IU', 'PET', 53.0233, 158.6499, 0.002],
    ['IU', 'DWPF', 28.1103, -81.4327, 0.162],
    ['IU', 'RSSD', 44.1212, -104.0359, 0.0692],
    ['IU', 'HNR', -9.4387, 159.9475, 0.1],
    ['IU', 'KNTN', -2.7744, -171.7186, 0.002],
    ['IU', 'WCI', 38.2289, -86.2939, 0.132],
    ['IU', 'KOWA', 14.4967, -4.014, 0.005],
    ['IU', 'PTCN', -25.0713, -130.0953, 0.002],
    ['IU', 'CHTO', 18.8141, 98.9443, 0.101],
    ['IU', 'ADK', 51.8823, -176.6842, 0.0],
    ['IU', 'RAR', -21.2125, -159.7733, 0.1],
    ['IU', 'TUC', 32.3098, -110.7847, 0.001],
    ['IU', 'OTAV', 0.2376, -78.4508, 0.015],
    ['IU', 'DAV', 7.0697, 125.5791, 0.001],
    ['IU', 'AFI', -13.9085, -171.7827, 0.001],
    ['IU', 'BBSR', 32.3713, -64.6963, 0.0314],
    ['IU', 'WVT', 36.1297, -87.83, 0.0],
    ['IU', 'SJG', 18.1091, -66.15, 0.0],
    ['IU', 'MIDW', 28.2156, -177.3698, 0.09359999999999999],
    ['IU', 'XMAS', 2.0448, -157.4457, 0.001],
    ['IU', 'RAO', -29.245, -177.929, 0.0005],
    ['IU', 'SSPA', 40.6358, -77.8876, 0.1],
    ['IU', 'CASY', -66.2792, 110.5354, 0.005],
    ['IU', 'KIP', 21.42, -158.0112, 0.033],
    ['IU', 'COR', 44.5855, -123.3046, 0.0],
    ['IU', 'POHA', 19.7573, -155.5326, 0.0803],
    ['IU', 'SFJ', 66.9967, -50.6156, 0.0],
    ['IU', 'TRIS', -37.0681, -12.3152, 0.002],
    ['IU', 'SLBS', 23.6858, -109.9443, 0.0],
    ['IU', 'ANTO', 39.868, 32.7934, 0.195],
    ['IU', 'GNI', 40.148, 44.741, 0.1],
    ['IU', 'LCO', -29.011, -70.7005, 0.0],
    ['IU', 'GRFO', 49.6909, 11.2203, 0.099],
    ['IU', 'TIXI', 71.6341, 128.8667, 0.0],
    ['IU', 'WAKE', 19.2834, 166.652, 0.08170000000000001],
    ['IU', 'TARA', 1.3549, 172.9229, 0.001],
    ['IU', 'KBS', 78.9154, 11.9385, 0.003],
    ['IU', 'PMSA', -64.7744, -64.0489, 0.0],
    ['IU', 'MAKZ', 46.808, 81.977, 0.01],
    ['IU', 'KONO', 59.6491, 9.5982, 0.34],
    ['IU', 'MA2', 59.5756, 150.77, 0.002],
    ['IU', 'LSZ', -15.2779, 28.1882, 0.0],
    ['IU', 'GUMO', 13.5893, 144.8684, 0.109],
    ['IU', 'BILL', 68.0653, 166.4531, 0.0],
    ['IU', 'QSPA', -89.9289, 144.4382, 0.27],
    ['IU', 'TEIG', 20.2263, -88.2763, 0.111],
    ['IU', 'PTGA', -0.7308, -59.9666, 0.096],
    ['IU', 'FUNA', -8.5259, 179.1966, 0.0],
    ['IU', 'KIEV', 50.7012, 29.2242, 0.04],
    ['IU', 'SDV', 8.8839, -70.634, 0.032],
    ['MN', 'TNV', -74.7, 164.12, 0.005],
    ['GE', 'PMG', -9.4092, 147.1539, 0.003],
    ['GE', 'SFJD', 66.996, -50.6215, 0.0],
    ['GE', 'LVC', -22.6182, -68.9113, 0.03],
    ['GE', 'KMBO', -1.1268, 37.2523, 0.02],
    ['GE', 'SFJ', 66.9967, -50.6156, 0.0],
    ['GE', 'TIRR', 44.4581, 28.4128, 0.0],
    ['GE', 'KBS', 78.9256, 11.9417, 0.0]]

# %%

sta_list.sort()
for sta in sta_list:
    print(sta)

# %%

latitudes = np.array([sta[2] for sta in sta_list])
longitudes = np.array([sta[3] for sta in sta_list])

# %%
# Bruteforce compute all distances
# Number of stations
N = len(sta_list)

dists = np.zeros((N, N))

for i in range(0, N-1):
    dists[i, i+1:] = locations2degrees(
        latitudes[i], longitudes[i],
        latitudes[i+1:], longitudes[i+1:])

# %%
# Add transpose
dists = dists + dists.T

# %%
# Wanted stations
wanted_stations = [
    ('G', 'ECH'),
    ('II', 'ABPO'),
    ('II', 'ALE'),
    ('II', 'ARU'),
    ('II', 'BFO'),
    ('II', 'KDAK'),
    ('IU', 'ANMO'),
    ('IU', 'HRV'),
    ('IU', 'KONO')]

# Get IU BFO
selected_indeces = np.array([], dtype=int)
for _i, sta in enumerate(sta_list):
    if (sta[0], sta[1]) in wanted_stations:
        selected_indeces = np.append(selected_indeces, _i)

if len(selected_indeces) == 0:
    raise ValueError("stations not found.")


# %%
plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax1.plot(longitudes, latitudes, 'o')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
ax2 = plt.subplot(2, 1, 2)

sc = ax2.scatter(
    longitudes,
    latitudes, s=4,
    c=np.min(dists[:, selected_indeces], axis=1),
    marker='o',
    vmin=20, vmax=90, cmap='rainbow')

plt.plot(longitudes[selected_indeces],
         latitudes[selected_indeces], 'o',
         markerfacecolor='none', markeredgecolor='red')

plt.colorbar(sc, orientation='horizontal')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show(block=False)

# %%
# Now select 70 points total
Nsel = 70

# %%

for i in range(len(selected_indeces), Nsel):

    distancesToEstablishedPoints = dists[:, selected_indeces]

    mindist_ToEstablishedPoints = np.min(
        distancesToEstablishedPoints, axis=1)

    kn = mindist_ToEstablishedPoints.argmax()

    selected_indeces = np.append(selected_indeces, kn)

# %%

plt.figure()

sc = plt.scatter(
    longitudes,
    latitudes, s=4,
    c=np.min(dists[:, selected_indeces], axis=1),
    marker='o',
    vmin=1, vmax=15, cmap='rainbow',
    label='Not selected')

plt.plot(longitudes, latitudes, 'o',
         markerfacecolor='none', markeredgecolor='black', label='All',
         markeredgewidth=0.75)
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.plot(longitudes[selected_indeces],
         latitudes[selected_indeces], 'o',
         markerfacecolor='black', markeredgecolor='none', label='subselection',
         markersize=4.5)
plt.legend(loc='lower left', frameon=False, fancybox=False,
           ncol=3, bbox_to_anchor=(0, 1), borderpad=0.5, borderaxespad=0.0, markerfirst=False)
cbar = plt.colorbar(sc, orientation='horizontal', shrink=0.7)
cbar.set_label('Distance to closest station [deg]')

plt.show(block=False)
plt.savefig('even_station_distribution.pdf')

# %%

for i in np.sort(selected_indeces):
    net, sta, _, _, _ = sta_list[i]
    print(f"{net}.{sta}")
