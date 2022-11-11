# %%
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os
import numpy as np
import time
x = np.random.randn(6, 5, 5, 5, 30, 4000)
x[::3].shape
x.flatten().shape
x[:, :, :, :, ::3, :] = np.arange(
    6*5*5*5*10*4000).reshape(6, 5, 5, 5, 10, 4000)
x.dtype
x = x.astype(np.float32)

# %%
compressors = [None, 'lzf', 'gzip']
compressor_opts = dict()
compressor_opts[None] = None
compressor_opts['gzip'] = 9
compressor_opts['lzf'] = None
compressor_opts['zlib'] = None

shuffle = [True, False]
chunks = [None,  1, 5, 10, 20]


timing = []
print(['shuffle', 'compressor', 'chunks', 'filesize', 'write', 'read-1', 'read-4'])
for _compressor in compressors:
    for _shuffle in shuffle:
        for _chunk in chunks:
            tch = None if _chunk is None else (6, 5, 5, 5, _chunk, 4000)
            lch = 'NN' if _chunk is None else str(_chunk).zfill(2)
            filename = f'/scratch/gpfs/lsawade/h5test.h5'

            with h5py.File(filename, 'w') as F:
                t0 = time.time()
                F.create_dataset(
                    'x', data=x,
                    chunks=tch,
                    compression=_compressor,
                    compression_opts=compressor_opts[_compressor],
                    shuffle=_shuffle)
                t1 = time.time()

            filesize = os.path.getsize(filename)

            with h5py.File(filename, 'r') as F:
                # Get one element
                t2 = time.time()
                xin = F['x'][:, :, :, :, 5, :]
                t3 = time.time()

                # Get set of elements
                t4 = time.time()
                xin = F['x'][:, :, :, :, np.arange(1, 8, 2, dtype=int), :]
                t5 = time.time()
                setup = [_shuffle, _compressor, _chunk,
                         filesize, t1-t0, t3-t2, t5-t4]
                print("Finalval: ", setup)
                timing.append(setup)

            os.remove(filename)


# %%

tarray = np.array(timing)

# %%
shufflebar = tarray[tarray[:, 0].astype(bool) == True, 1:]
noshuffle = tarray[tarray[:, 0].astype(bool) == False, 1:]

# %%
plt.figure(figsize=(10, 8))
counter = 1
Ncomp = len(compressors)
Nchunk = len(chunks)
width = 1.0
xwidth = 2.0
x = xwidth*np.arange(Nchunk)

barwidth = width/Ncomp
baroffset = barwidth/Ncomp

labels = ['Filesize in MB', 'Write-Time [s]',
          '1 el. Read [s]', '4 el. Read [s]']

chunkslabels = chunks
chunkslabels[0] = 'Auto'

# ax = plt.subplot(3, 2, counter)
for i in range(4):

    ax = plt.subplot(4, 2, (2*i)+1)

    for j in range(Ncomp):
        values = shufflebar[shufflebar[:, 0] == compressors[j], 1:]
        print(x + j*width/3 - width/2)
        print(values[:, i+1])
        if i == 0:
            values[:, i+1] /= 1000000.0
        rects1 = ax.bar(x + j*barwidth + baroffset - xwidth/4, values[:, i+1],
                        barwidth, label=str(compressors[j]), lw=0.0)

    plt.ylabel(labels[i])
    ax.set_xticks(x, chunkslabels)

    if i == 0:
        plt.title('Shuffle')
        ax.legend(frameon=False, loc='upper center', ncol=3)

    # Log scale for writing
    if i == 1 or i == 2:
        ax.set_yscale('log')

    if i != 3:
        ax.tick_params(labelbottom=False)

    ax = plt.subplot(4, 2, (2*i)+2, sharex=ax, sharey=ax)

    for j in range(Ncomp):
        values = noshuffle[shufflebar[:, 0] == compressors[j], 1:]
        print(values[:, i+1])
        if i == 0:
            values[:, i+1] /= 1000000.0

        rects1 = ax.bar(x + j*barwidth + baroffset - xwidth/4, values[:, i+1],
                        barwidth, label=compressors[j], lw=0.0)

    ax.set_xticks(x, chunkslabels)
    ax.set_xlabel('Chunks')
    ax.tick_params(labelleft=False)
    if i != 3:
        ax.tick_params(labelbottom=False)

    if i == 0:
        plt.title('No Shuffle')

plt.savefig('testbench.pdf')
