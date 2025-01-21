import h5py
import numpy as np
from sklearn.decomposition import PCA
from myphdlib.general.toolkit import psth2

def getTimeRange(
    spikeTimestamps,
    spikeClusters=None,
    targetCluster=None,
    pad=5,
    ):
    """
    """

    if targetCluster is None:
        spikeIndices = np.arange(spikeTimestamps)
    else:
        spikeIndices = np.where(spikeClusters == targetCluster)[0]
    t1 = np.floor(spikeTimestamps[spikeIndices].min()) + pad
    t2 = np.ceil(spikeTimestamps[spikeIndices].max()) - pad

    return t1, t2

def loadMlatiContinuous(
    filename,
    binsize=0.05,
    nlags=(10, 10),
    maximumProbabilityValue=0.001,
    ):
    """
    """

    # Load datasets
    with h5py.File(filename, 'r') as stream:
        eyePosition = np.array(stream['pose/filtered'])[:, 0]
        nFramesRecorded = len(eyePosition)
        frameTimestamps = np.array(stream['frames/left/timestamps'])[:nFramesRecorded]
        spikeTimestamps = np.array(stream[f'spikes/timestamps'])
        spikeClusters = np.array(stream[f'spikes/clusters'])
        pValues = np.vstack([
            np.array(stream['zeta/saccade/nasal/p']),
            np.array(stream['zeta/saccade/temporal/p'])
        ]).min(0)

    #
    uniqueClusters = np.unique(spikeClusters)
    if maximumProbabilityValue is None:
        targetClusters = uniqueClusters
    else:
        clusterIndices = np.arange(len(uniqueClusters))[pValues < maximumProbabilityValue]
        targetClusters = uniqueClusters[clusterIndices]
    nUnits = len(targetClusters)

    #
    t1, t2 = getTimeRange(filename)
    binEdges = np.arange(t1, t2 + binsize, binsize)
    nBins = binEdges.size - 1
    R = np.full([nBins, nUnits], np.nan, dtype=np.float16)

    # Main loop
    for j, spikeCluster in enumerate(targetClusters):

        # Compute histograms
        counts, edges = np.histogram(
            spikeTimestamps[spikeClusters == spikeCluster],
            bins=binEdges
        )
        fr = counts / binsize
        R[:, j] = fr

    #
    nLags = sum(nlags) + 1
    binOffsets = np.arange(-1 * nlags[0], nlags[1] + 1, 1)
    X = np.full([nBins, nUnits * nLags], np.nan, dtype=np.float16)
    for i, iBin in enumerate(range(nBins)):
        end = '\r' if iBin + 1 != nBins else '\n'
        print(f'Populating response matrix for time bin {iBin + 1} out of {nBins} ({(iBin + 1) / nBins * 100:.1f}%)', end=end)
        for j, iUnit in enumerate(range(nUnits)):
            for k, binOffset in enumerate(binOffsets):
                if iBin + binOffset < 0:
                    fr = 0.0
                elif iBin + binOffset > nBins - 1:
                    fr = 0.0
                else:
                    fr = R[iBin + binOffset, iUnit]
                X[iBin, k * nUnits + iUnit] = fr

    #
    t1, t2 = getTimeRange(spikeTimestamps)
    tRaw = frameTimestamps[:-1] + (np.diff(frameTimestamps) / 2)

    #
    vRaw = np.diff(eyePosition)
    vRaw[np.isnan(vRaw)] = np.interp(tRaw[np.isnan(vRaw)], tRaw, vRaw) # Impute with interpolation
    leftEdges = np.arange(t1, t2, binsize)
    rightEdges = leftEdges + binsize
    binEdges = np.vstack([leftEdges, rightEdges]).T
    y = np.interp(binEdges.mean(1), tRaw, vRaw, left=np.nan, right=np.nan).reshape(-1, 1)

    return R, X, y

def loadMlati(
    filename,
    binsize=0.01,
    eventWindow=(-0.1, 0.1),
    laggedBins=(50, 50),
    maximumProbabilityValue=0.001,
    minimumFiringRate=0.2,
    standardizeFiringRate=False,
    spikeTimestamps=None,
    spikeClusters=None
    ):
    """
    """

    with h5py.File(filename, 'r') as stream:
        eyePosition = np.array(stream['pose/filtered'])[:, 0]
        nFramesRecorded = len(eyePosition)
        frameTimestamps = np.array(stream['frames/left/timestamps'])[:nFramesRecorded]
        if spikeTimestamps is None:
            spikeTimestamps = np.array(stream[f'spikes/timestamps'])
        if spikeClusters is None:
            spikeClusters = np.array(stream[f'spikes/clusters'])
        pValues = np.vstack([
            np.array(stream['zeta/saccade/nasal/p']),
            np.array(stream['zeta/saccade/temporal/p'])
        ]).min(0)
        saccadeTimestamps = np.array(stream['saccades/predicted/left/timestamps'])
        saccadeLabels = np.array(stream['saccades/predicted/left/labels'])

    #
    tRaw = frameTimestamps[:-1] + (np.diff(frameTimestamps) / 2)
    vRaw = np.diff(eyePosition)
    vRaw[np.isnan(vRaw)] = np.interp(tRaw[np.isnan(vRaw)], tRaw, vRaw) # Impute with interpolation

    #
    y = list()
    tEval = np.arange(eventWindow[0], eventWindow[1], binsize) + (binsize / 2)
    for saccadeTimestamp in saccadeTimestamps[:, 0]:
        wf = np.interp(
            tEval + saccadeTimestamp,
            tRaw,
            vRaw
        )
        y.append(wf)
    y = np.array(y)

    # Exclude units without event-related activity
    uniqueClusters = np.unique(spikeClusters)
    if maximumProbabilityValue is None:
        targetClusters = uniqueClusters
    else:
        clusterIndices = np.arange(len(uniqueClusters))[pValues <= maximumProbabilityValue]
        targetClusters = uniqueClusters[clusterIndices]

    # Exclude units with too low of a firing rate (probabily partial units)
    if minimumFiringRate is not None:
        unitIndices = list()
        for iUnit, targetCluster in enumerate(targetClusters):
            t1, t2 = getTimeRange(
                spikeTimestamps,
                spikeClusters,
                targetCluster,
                pad=5   
            )
            spikeIndices = np.where(spikeClusters == targetCluster)[0]
            nBins = int((t2 - t1) / binsize)
            nSpikes, binEdges_ = np.histogram(
                spikeTimestamps[spikeIndices],
                range=(t1, t2),
                bins=nBins
            )
            xb = nSpikes.mean() / binsize
            if xb < minimumFiringRate:
                unitIndices.append(iUnit)
        targetClusters = np.delete(targetClusters, unitIndices)
    nUnits = len(targetClusters)

    # Compute the edges of the time bins centered on the saccade
    indexOffsets = np.arange(-1 * laggedBins[0], laggedBins[1] + 1, 1)
    binCenters = indexOffsets * binsize
    leftEdges = np.around(binCenters - (binsize / 2), 5)
    rightEdges = leftEdges + binsize
    binEdges = np.concatenate([leftEdges, rightEdges[-1:]])

    # Compute histograms and store in response matrix of shape N units x M saccades x P time bins
    R = list()
    for iUnit, targetCluster in enumerate(targetClusters):
        end = '\r' if iUnit + 1 != nUnits else '\n'
        print(f'Computing histograms for unit {iUnit + 1} out of {nUnits} ...', end=end)
        spikeIndices = np.where(spikeClusters == targetCluster)[0]
        sample = list()
        for saccadeTimestamp in saccadeTimestamps[:, 0]:
            nSpikes, binEdges_ = np.histogram(
                spikeTimestamps[spikeIndices],
                bins=np.around(binEdges + saccadeTimestamp, 4)
            )
            fr = nSpikes / binsize
            sample.append(fr)
        R.append(sample)
    R = np.array(R)

    # Standardize the firing rate
    if standardizeFiringRate:
        for iUnit, targetCluster in enumerate(targetClusters):
            t1, t2 = getTimeRange(
                spikeTimestamps,
                spikeClusters,
                targetCluster,
                pad=5
            )
            spikeIndices = np.where(spikeClusters == targetCluster)[0]
            nBins = int((t2 - t1) / binsize)
            nSpikes, binEdges_ = np.histogram(
                spikeTimestamps[spikeIndices],
                range=(t1, t2),
                bins=nBins
            )
            xb = nSpikes.mean() / binsize
            sd = nSpikes.std() / binsize
            R[iUnit] = ((R[iUnit] - xb) / sd)

    # Populate response matrix
    X = list()
    nSaccades = saccadeTimestamps.shape[0]
    for iSaccade in range(nSaccades):
        sample = list()
        for offset in indexOffsets:  # Outer loop over lags
            for iUnit in range(nUnits):  # Inner loop over neurons
                binIndex = int((R.shape[2] - 1) / 2) + offset
                fr = R[iUnit, iSaccade, binIndex]
                sample.append(fr)
        X.append(sample)
    X = np.array(X)

    # Return the direction of each saccade
    z = saccadeLabels.reshape(-1, 1)

    # Remove samples with NaN values
    mask = np.vstack([
        np.isnan(X).any(1),
        np.isnan(y).any(1),
        np.isnan(z).any(1)
    ]).any(0)
    X = np.delete(X, mask, axis=0)
    y = np.delete(y, mask, axis=0)
    z = np.delete(z, mask, axis=0)

    return R, X, y, z, tEval, binCenters

def generateFakeSpikes(
    saccadeTimestamps,
    nUnits=10,
    responseLatency=-0.05,
    responseVariability=0.03,
    responseProbability=0.7,
    minimumSpikeCount=0,
    maximumSpikeCount=10,
    baselineFiringRate=2,
    minimumInterSpikeInterval=0.005,
    dropoutRate=0.1,
    pad=3
    ):
    """
    """

    spikeTimestamps = list()
    spikeClusters = list()
    saccadeTimestampsFiltered = saccadeTimestamps[np.invert(np.isnan(saccadeTimestamps[:, 0])), 0]
    t1 = saccadeTimestampsFiltered.min() - pad
    t2 = saccadeTimestampsFiltered.max() + pad
    for iUnit in range(nUnits):

        end = '\r' if iUnit + 1 != nUnits else '\n'
        print(f'Generating spikes for unit {iUnit + 1} out of {nUnits}', end=end)

        #
        t = list()
        c = list()

        # Random activity
        nSpikes = int(round((t2 - t1) * baselineFiringRate, 0))
        for spikeTimestamp in np.random.uniform(low=t1, high=t2, size=nSpikes):
            t.append(spikeTimestamp)
            c.append(iUnit + 1)

        # Units with no event-related activity
        dropUnit = np.random.choice([True, False], p=[dropoutRate, 1 - dropoutRate])
        if dropUnit:
            for spikeTimestamp in t:
                spikeTimestamps.append(spikeTimestamp)
            for spikeCluster in c:
                spikeClusters.append(spikeCluster)
            continue

        # Event-related activity
        for saccadeTimestamp in saccadeTimestampsFiltered:
            if np.isnan(saccadeTimestamp):
                continue
            responseElicited = np.random.choice([True, False], p=[1 - responseProbability, responseProbability])
            if responseElicited:
                nSpikes = int(np.random.choice(np.arange(minimumSpikeCount, maximumSpikeCount + 1)))
                for spikeTimestamp in np.random.normal(loc=responseLatency, scale=responseVariability, size=nSpikes):

                    # Enforce refractory period
                    dt = np.min(np.abs(np.array(t) - spikeTimestamp))
                    if dt < minimumInterSpikeInterval:
                        continue
                    
                    t.append(round(spikeTimestamp + saccadeTimestamp, 3))
                    c.append(iUnit + 1)
        
        #
        for spikeTimestamp in t:
            spikeTimestamps.append(spikeTimestamp)
        for spikeCluster in c:
            spikeClusters.append(spikeCluster)

    spikeTimestamps, spikeClusters = np.array(spikeTimestamps), np.array(spikeClusters)
    index = np.argsort(spikeTimestamps)
    return spikeTimestamps[index], spikeClusters[index]