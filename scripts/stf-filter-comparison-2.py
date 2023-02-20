import os
from numpy import *
from matplotlib.pyplot import *
from gf3d.stf import create_stf

# File paths
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
FIGUREDIR = os.path.dirname(SCRIPTDIR)


t0 = -120
tc = 0.0
dt = 0.25
nt = 961


hdur = 8.0
# cutoff = 10

# t = arange(t0, t0+nt*dt, dt)

cutoffs = [1/8.00, 1/10.0, 1/12.0, 1/14.0]

figure(figsize=(9, 9))

for i in range(4):

    subplot(2, 2, 1+i)
    t, stf = create_stf(t0, tc, nt, dt, hdur)
    _, stf_bessel = create_stf(
        t0, tc, nt, dt, hdur, cutoff=cutoffs[i], lpfilter='bessel')
    _, stf_butter = create_stf(
        t0, tc, nt, dt, hdur, cutoff=cutoffs[i], lpfilter='butter')
    _, stf_cheby1 = create_stf(
        t0, tc, nt, dt, hdur, cutoff=cutoffs[i], lpfilter='cheby1')
    _, stf_cheby2 = create_stf(
        t0, tc, nt, dt, hdur, cutoff=cutoffs[i], lpfilter='cheby2')

    plot(t, stf, label='Plain')
    plot(t, stf_bessel, label="Bessel")
    plot(t, stf_butter, label="Butter")
    plot(t, stf_cheby1, label="Cheby1")
    plot(t, stf_cheby2, label="Cheby2")
    xlim(min(t), max(t))
    title(f'Cutoff {1/cutoffs[i]:.2f} s')
    ax = gca()
    if i <= 1:
        ax.tick_params(labelbottom=False)
    else:
        xlabel('Time [s]')

    if i % 2 == 1:
        ax.tick_params(labelleft=False)
    else:
        ylabel('A')

legend(frameon=False)

suptitle(f'Comparing filters: dt = {dt:.3f} s, hdur = {hdur:.3f} s')

savefig(os.path.join(FIGUREDIR, 'stf-comparison-2.pdf'), dpi=300)

##########################################################################
##########################################################################
##########################################################################

lpfilters = ['bessel', 'butter', 'cheby1', 'cheby2']
cutoffs = [1/8.001, 1/10.0, 1/12.0, 1/14.0]
# cutoffs = [2.0, 2.25, 2.5, 2.75]  # [1.0, 1.25, 1.5, 1.75]

figure(figsize=(7, 5))

for _i, _lpfilter in enumerate(lpfilters):
    subplot(2, 2, 1+_i)

    for cutoff in cutoffs:
        t, stf = create_stf(
            t0, tc, nt, dt, hdur, cutoff=cutoff, lpfilter=_lpfilter)

        plot(t, stf, label=f"{1/cutoff:.2f} s", lw=0.5)

    xlim(min(t), max(t))
    title(f'{_lpfilter.capitalize():s}')
    legend(frameon=False)

    ax = gca()

    if _i <= 1:
        ax.tick_params(labelbottom=False)
    else:
        xlabel('Time [s]')

    if _i % 2 == 1:
        ax.tick_params(labelleft=False)
    else:
        ylabel('A')


suptitle(f'Comparing filters: dt = {dt:.3f} s, hdur = {hdur:.3f} s')

savefig(os.path.join(FIGUREDIR, 'stf-filters-1-by-1.pdf'), dpi=300)
