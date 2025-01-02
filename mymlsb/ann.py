import torch
import numpy as np
from torch import nn
from torch import optim
from collections import OrderedDict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# TODO
# [ ] Implement an averaging function with a sliding window to watch test MSE
    
class _DeepNeuralNetwork(nn.Module):
    """
    Neural network with multiple hidden layers
    """

    def __init__(self, inputLayerSize, hiddenLayerSizes=[10,]):
        """
        """

        super().__init__()
        layers = list()
        layers.append(nn.Linear(inputLayerSize, hiddenLayerSizes[0]))
        layers.append(nn.ReLU())
        nLayers = len(hiddenLayerSizes)
        for iLayer in range(nLayers):
            s1 = hiddenLayerSizes[iLayer]
            if iLayer + 1 < nLayers:
                s2 = hiddenLayerSizes[iLayer + 1]
                layers.append(nn.Linear(s1, s2))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(s1, 1))
        self.seq = nn.Sequential(*layers)
            
        return
    
    def forward(self, x):
        """
        """

        return self.seq(x)
    
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    """
    """

    def __init__(
        self,
        inputLayerSize,
        hiddenLayerSizes=[100,],
        lr=0.001,
        nEpochs=1000000,
        minimumDelta=0,
        patience=100,
        slidingWindowSize=100,
        device=None
        ):
        """
        """

        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSizes = hiddenLayerSizes
        self.nEpochs = nEpochs
        self.lr = lr
        self.slidingWindowSize = slidingWindowSize
        self.minimumDelta = minimumDelta
        self.patience = patience
        self.ann = None
        self.performance = None
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        return
    
    def fit(self, X, y):
        """
        """

        #
        if len(y.shape) == 1 or y.shape[0] != X.shape[0]:
            raise Exception('y must have the same size as X along the first dimension')

        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y, dtype=torch.float32).to(self.device)
        self.ann = _DeepNeuralNetwork(
            self.inputLayerSize,
            self.hiddenLayerSizes
        ).to(self.device)
        nSamples = X.shape[0]
        trainIndex = np.arange(int(round(nSamples / 5 * 4)))
        testIndex = np.arange(trainIndex[-1] + 1, nSamples)
        lossFunction = nn.MSELoss()
        optimizer = optim.Adam(self.ann.parameters(), lr=self.lr)
        counter = 0
        self.performance = np.full(self.nEpochs, np.nan)
        lossPreviousAverage = np.nan, np.nan
        lossMinimum = np.inf
        stateDict = None
        windowData = np.full(self.slidingWindowSize, np.nan)
        bestEpoch = None

        # Main training loop
        try:
            for iEpoch in range(self.nEpochs):

                # Training step
                self.ann.train()
                predictions = self.ann(Xt[trainIndex])
                lossObject = lossFunction(predictions, yt[trainIndex])
                optimizer.zero_grad()
                lossObject.backward()
                optimizer.step()

                # Validation step
                self.ann.eval()
                with torch.no_grad():
                    predictions = self.ann(Xt[testIndex])
                    lossObject = lossFunction(predictions, yt[testIndex])
                    lossCurrentEpoch = lossObject.item()

                # Keep track of the best cross-validated model
                if lossCurrentEpoch < lossMinimum:
                    bestEpoch = iEpoch
                    stateDict = self.ann.state_dict()
                    lossMinimum = lossCurrentEpoch

                # Compute average loss over window
                i = np.take(np.arange(self.slidingWindowSize), iEpoch, mode='wrap')
                windowData[i] = lossCurrentEpoch
                lossCurrentAverage = np.nanmean(windowData)
                self.performance[iEpoch] = lossCurrentAverage
                lossDeltaAverage = lossPreviousAverage - lossCurrentAverage

                # Skip further evaluation for the first iteration
                if iEpoch < self.slidingWindowSize:
                    lossPreviousAverage = lossCurrentAverage
                    continue

                # Report performance
                end = '\r' if iEpoch + 1 != self.nEpochs else '\n'
                print(f'Epoch {iEpoch + 1} out of {self.nEpochs}: loss={lossCurrentAverage:4f} (delta={lossDeltaAverage:.9f})', end=end)

                # Stop early if change in loss is slowing down or worsening (to prevent overfitting)
                if self.minimumDelta is not None:

                    # Performance slowdown is allowed to reset if improvement is detected
                    if lossDeltaAverage < self.minimumDelta:
                        counter += 1
                    else:
                        counter = 0
                    if counter >= self.patience:
                        print(f'Epoch {iEpoch + 1} out of {self.nEpochs}: loss={lossCurrentAverage:4f} (delta={lossDeltaAverage:.9f})', end='\n')
                        break

                # Update the previous loss variable for the next iteration
                lossPreviousAverage = lossCurrentAverage
            
        #
        except KeyboardInterrupt:
            pass

        # Load the best cross-validated model
        print(f'Best performance recorded on the {bestEpoch + 1} epoch')
        self.ann.load_state_dict(stateDict)

        return self
    
    def predict(self, X):
        """
        """

        tensor = self.ann(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu()
        return np.array(tensor.detach())
    
def trainModelWithGridSearch(X, y, n=5, delta=0.0001, nEpochs=1000):
    """
    """

    xSplit = np.array_split(X, n, axis=0)
    xTrain = np.concatenate(xSplit[:-1], axis=0)
    xTest = xSplit[-1]
    ySplit = np.array_split(y, n, axis=0)
    yTrain = np.concatenate(ySplit[:-1], axis=0)
    yTest = ySplit[-1]

    regressor = PyTorchRegressor(
        inputLayerSize=X.shape[1],
        nEpochs=nEpochs,
        delta=delta
    )
    cv = TimeSeriesSplit(n_splits=n)
    params = {
        'hiddenLayerSizes': [[1,],    [1, 1, 1,],          [1, 1, 1, 1, 1],
                             [100,],  [100, 100, 100,],    [100, 100, 100, 100, 100],
                             [1000,], [1000, 1000, 1000,], [1000, 1000, 1000, 1000, 1000],
                            ],
    }
    gs = GridSearchCV(
        regressor,
        params,
        cv=cv,
        verbose=True,
        scoring='neg_mean_squared_error'
    )
    gs.fit(xTrain, yTrain)
    regressor = gs.best_estimator_
    yPredicted = regressor.predict(xTest)
    mse = np.mean(np.power(yPredicted - yTest, 2))
    print(f'Best estimator identified with MSE of {mse:.3f}')

    return gs, yTest, yPredicted