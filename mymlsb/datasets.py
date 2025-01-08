import h5py
import numpy as np
from sklearn.decomposition import PCA
from myphdlib.general.toolkit import psth2

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

def loadMlatiForRidge(
    filename,
    binsize=0.01,
    windowForSpikes=(-0.1, 0.1),
    windowForVelocity=(-0.1, 0.1),
    maximumProbabilityValue=None,
    ):
    """
    Compute firing rate for all units at a range of time lags (X). Eye velocity
    (y) is sampled from around the time of saccade onset.
    """

    # Load datasets
    with h5py.File(filename, 'r') as stream:
        eyePosition = np.array(stream[f'pose/filtered'])[:, :2]
        frameTimestamps = np.array(stream[f'frames/left/timestamps'])
        saccadeTimestamps = np.array(stream[f'saccades/predicted/left/timestamps'])[:, 0]
        spikeTimestamps = np.array(stream[f'spikes/timestamps'])
        spikeClusters = np.array(stream[f'spikes/clusters'])
        pValues = np.vstack([
            np.array(stream['zeta/saccade/nasal/p']),
            np.array(stream['zeta/saccade/temporal/p'])
        ]).min(0)

    # Compute velocity
    tRaw = frameTimestamps[0] + np.cumsum(np.diff(frameTimestamps))
    vRaw = np.diff(eyePosition, axis=0)
    for j in range(vRaw.shape[1]):
        i = np.where(np.isnan(vRaw[:, j]))[0]
        vRaw[i, j] = np.interp(tRaw[i], tRaw, vRaw[:, j]) # Impute with interpolation

    #
    uniqueClusters = np.unique(spikeClusters)
    if maximumProbabilityValue is None:
        targetClusters = uniqueClusters
    else:
        clusterIndices = np.arange(len(uniqueClusters))[pValues < maximumProbabilityValue]
        targetClusters = uniqueClusters[clusterIndices]

    #
    X = list()
    y = list()
    binCenters, M = psth2(np.array([0,]), np.array([0,]), window=windowForVelocity, binsize=binsize)
    n = saccadeTimestamps.size * len(binCenters) * len(targetClusters)
    counter = 0
    for saccadeTimestamp in saccadeTimestamps:
        for dt in binCenters:
            Xi = np.array([])
            yi = np.array([
                np.interp(saccadeTimestamp + dt, tRaw, vRaw[:, 0]),
                np.interp(saccadeTimestamp + dt, tRaw, vRaw[:, 1]),
            ])
            for spikeCluster in targetClusters:
                end = '\r' if counter + 1 != n else '\n'
                percent = counter / n * 100
                print(f'Working on combination {counter + 1} out of {n} ... ({percent:.1f}%)', end=end)
                spikeIndices = np.where(spikeClusters == spikeCluster)[0]
                t, M = psth2(
                    np.array([saccadeTimestamp + dt,]),
                    spikeTimestamps[spikeIndices],
                    window=windowForSpikes,
                    binsize=binsize
                )
                Xi = np.concatenate([Xi, M.flatten() / binsize])
                counter += 1
            X.append(Xi)
            y.append(yi)
    
    #
    X = np.array(X)
    y = np.array(y)

    return X, y

def loadMlatiWithPCA(
    filename,
    binsize=0.01,
    windowForSpikes=(-0.1, 0.1),
    windowForSaccades=(-0.1, 0.1),
    maximumProbabilityValue=None,
    ):
    """
    """

    # Load datasets
    with h5py.File(filename, 'r') as stream:
        eyePosition = np.array(stream[f'pose/filtered'])[:, :2]
        frameTimestamps = np.array(stream[f'frames/left/timestamps'])
        saccadeTimestamps = np.array(stream[f'saccades/predicted/left/timestamps'])[:, 0]
        spikeTimestamps = np.array(stream[f'spikes/timestamps'])
        spikeClusters = np.array(stream[f'spikes/clusters'])
        pValues = np.vstack([
            np.array(stream['zeta/saccade/nasal/p']),
            np.array(stream['zeta/saccade/temporal/p'])
        ]).min(0)

    # Compute velocity
    tRaw = frameTimestamps[0] + np.cumsum(np.diff(frameTimestamps))
    vRaw = np.diff(eyePosition, axis=0)
    for j in range(vRaw.shape[1]):
        i = np.where(np.isnan(vRaw[:, j]))[0]
        vRaw[i, j] = np.interp(tRaw[i], tRaw, vRaw[:, j]) # Impute with interpolation

    #
    uniqueClusters = np.unique(spikeClusters)
    if maximumProbabilityValue is None:
        targetClusters = uniqueClusters
    else:
        clusterIndices = np.arange(len(uniqueClusters))[pValues < maximumProbabilityValue]
        targetClusters = uniqueClusters[clusterIndices]

    #
    t1 = np.floor(tRaw.min())
    t2 = np.ceil(tRaw.max())
    bins = np.arange(t1, t2 + binsize, binsize)
    xFull = np.full([bins.size - 1, len(targetClusters)], np.nan, dtype=np.float32)
    for j, spikeCluster in enumerate(targetClusters):
        counts, edges = np.histogram(
            spikeTimestamps[spikeClusters == spikeCluster],
            bins=bins
        )
        fr = (counts - counts.mean()) / counts.std()
        xFull[:, j] = fr
    xReduced = PCA(n_components=1).fit_transform(xFull)
    del xFull
    tReduced = bins[:-1] + ((bins[1] - bins[0]) / 2)
    return xReduced, tReduced

    # TODO: Figure out why activity along PC1 doesn't show visual response

    #
    X = list()
    y = list()
    binsForVelocity, M = psth2(np.array([0,]), np.array([0,]), window=windowForSaccades, binsize=binsize)
    binsForSpikes, M = psth2(np.array([0,]), np.array([0,]), window=windowForSpikes, binsize=binsize)
    n = saccadeTimestamps.size * binsForVelocity.size * binsForSpikes.size
    counter = 0
    for saccadeTimestamp in saccadeTimestamps:
        for dtVelocity in binsForVelocity:
            yi = np.array([
                np.interp(saccadeTimestamp + dtVelocity, tRaw, vRaw[:, 0]),
                np.interp(saccadeTimestamp + dtVelocity, tRaw, vRaw[:, 1]),
            ])
            Xi = list()
            for dtSpikes in binsForSpikes:
                end = '\r' if counter + 1 != n else '\n'
                percent = counter / n * 100
                print(f'Working on combination {counter + 1} out of {n} ... ({percent:.1f}%)', end=end)
                ti = saccadeTimestamp + dtVelocity + dtSpikes
                Xi.append(np.interp(ti, tReduced, xReduced.flatten()).item())
                counter += 1
            Xi = np.array(Xi)
            X.append(Xi)
            y.append(yi)
    X = np.array(X)
    y = np.array(y)

    return X, y

def getTimeRange(
    filename,
    ):
    """
    """

    with h5py.File(filename, 'r') as stream:
        spikeTimestamps = np.array(stream['spikes/timestamps'])
    t1 = np.floor(spikeTimestamps.min())
    t2 = np.ceil(spikeTimestamps.max())

    return t1, t2

def loadEyeVelocity(
    filename,
    ):
    """
    """

    return

def loadGratingContrast(
    filename,
    binsize=0.005,
    riseTime=0.001,
    probeDuration=0.05,
    ):
    """
    """

    #
    t1, t2 = getTimeRange(filename)
    t = np.arange(t1, t2, binsize) + (binsize / 2)

    #
    with h5py.File(filename, 'r') as stream:
        probeTimestamps = np.array(stream['stimuli/dg/probe/timestamps'])
    
    #
    xp = np.concatenate([
        probeTimestamps - riseTime,
        probeTimestamps,
        probeTimestamps + probeDuration,
        probeTimestamps + probeDuration + riseTime
    ])
    index = np.argsort(xp)
    xp = xp[index]
    fp = np.concatenate([
        np.zeros(probeTimestamps.size),
        np.ones(probeTimestamps.size),
        np.ones(probeTimestamps.size),
        np.zeros(probeTimestamps.size),
    ])
    fp = fp[index]

    return t, np.interp(t, xp, fp)

def loadNeuralData(
    filename,
    binsize=0.005,
    maximumProbabilityValue=0.001,
    ):
    """
    """

    # Load datasets
    with h5py.File(filename, 'r') as stream:
        spikeTimestamps = np.array(stream[f'spikes/timestamps'])
        spikeClusters = np.array(stream[f'spikes/clusters'])
        pValues = np.vstack([
            np.array(stream['zeta/probe/left/p']),
            np.array(stream['zeta/probe/right/p'])
        ]).min(0)

    #
    uniqueClusters = np.unique(spikeClusters)
    if maximumProbabilityValue is None:
        targetClusters = uniqueClusters
    else:
        clusterIndices = np.arange(len(uniqueClusters))[pValues < maximumProbabilityValue]
        targetClusters = uniqueClusters[clusterIndices]

    #
    t1, t2 = getTimeRange(filename)
    bins = np.arange(t1, t2 + binsize, binsize)
    t = bins[:-1] + (binsize / 2)
    xFull = np.full([bins.size - 1, len(targetClusters)], np.nan, dtype=np.float32)
    for j, spikeCluster in enumerate(targetClusters):
        counts, edges = np.histogram(
            spikeTimestamps[spikeClusters == spikeCluster],
            bins=bins
        )
        # fr = (counts - counts.mean()) / counts.std()
        fr = counts / binsize
        xFull[:, j] = fr

    return t, xFull

def lag(
    X,
    y,
    t,
    lags=[0,],
    nSamples=100,
    eventTimestamps=None,
    eventWindow=[-0.1, 0.1],
    ):
    """
    """

    #
    if eventTimestamps is None:
        tEval = np.linspace(
            low=t.min(),
            high=t.max(),
            size=nSamples + 2
        )[1:-1]
    else:
        tEval = list()
        eventIndices = np.random.choice(np.arange(eventTimestamps.size), size=nSamples, replace=True)
        eventIndices.sort()
        for eventTimestamp in eventTimestamps[eventIndices]:
            if np.isnan(eventTimestamp):
                continue
            t1 = eventTimestamp + eventWindow[0]
            t2 = eventTimestamp + eventWindow[1]
            tEval.append(np.random.uniform(
                low=t1,
                high=t2,
                size=1
            ).item())
        tEval = np.array(tEval)

    #
    xLag = np.full([nSamples, len(lags)], np.nan)
    yLag = np.full([nSamples, 1], np.nan)

    #
    for i, ti in enumerate(tEval):
        end = '\r' if (i + 1) < nSamples else '\n'
        print(f'Generating sample {i + 1} out of {nSamples} ...', end=end)
        for j, lag in enumerate(lags):
            xLag[i, j] = np.interp(
                ti + lag,
                t,
                X.flatten()
            )
        yLag[i, 0] = np.around(np.interp(
            ti,
            t,
            y.flatten()
        ), 0).item()

    return xLag, yLag, tEval   