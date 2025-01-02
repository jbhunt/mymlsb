import h5py
import numpy as np

def loadMlati(
    filename,
    binsizeVelocity=0.005,
    binsizeSpiking=0.1,
    offsetSpinking=0,
    trange=(None, None),
    minimumFiringRate=0.1,
    ):
    """
    """

    with h5py.File(filename, 'r') as stream:
        eyePosition = np.array(stream['pose/filtered'])[:, 0]
        frameTimestamps = np.array(stream['frames/left/timestamps'])
        spikeTimestamps = np.array(stream['spikes/timestamps'])
        spikeClusters = np.array(stream['spikes/clusters'])

    #
    tRaw = frameTimestamps[0] + np.cumsum(np.diff(frameTimestamps))
    if trange[0] is None:
        tmin = np.floor(tRaw.min())
    else:
        tmin = trange[0]
    if trange[1] is None:
        tmax = np.ceil(tRaw.max())
    else:
        tmax = trange[1]


    #
    vRaw = np.diff(eyePosition)
    vRaw[np.isnan(vRaw)] = np.interp(tRaw[np.isnan(vRaw)], tRaw, vRaw) # Impute with interpolation
    leftEdges = np.arange(tmin, tmax, binsizeVelocity)
    rightEdges = leftEdges + binsizeVelocity
    binEdges = np.vstack([leftEdges, rightEdges]).T
    y = np.interp(binEdges.mean(1), tRaw, vRaw, left=np.nan, right=np.nan).reshape(-1, 1)

    #
    X = list()
    nUnits = len(np.unique(spikeClusters))
    for iUnit, cluster in enumerate(np.unique(spikeClusters)):
        end = '\r' if iUnit + 1 != nUnits else '\n'
        print(f'Working on unit {iUnit + 1} out of {nUnits} ...', end=end)
        spikeIndices = np.where(spikeClusters == cluster)[0]
        t = spikeTimestamps[spikeIndices]
        x = list()
        for t2 in leftEdges:
            t2 -= offsetSpinking
            t1 = t2 - binsizeSpiking
            fr = np.sum(np.logical_and(t >= t1, t < t2)) / binsizeSpiking
            x.append(fr)
        if np.mean(x) < minimumFiringRate:
            continue
        X.append(x)

    #
    X = np.array(X).T
    X = np.delete(X, np.isnan(y).flatten(), axis=0)
    y = np.delete(y, np.isnan(y).flatten(), axis=0)

    return X, y