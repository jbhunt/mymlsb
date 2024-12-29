import torch
import numpy as np
from torch import nn
from torch import optim
from collections import OrderedDict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    
class DeepNeuralNetwork(nn.Module):
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
        hiddenLayerSizes=[10,],
        lr=0.001,
        nEpochs=1000000,
        earlyStoppingDelta=None,
        earlyStoppingPatience=100,
        maximumNegativeLoss=1000,
        ):
        """
        """

        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSizes = hiddenLayerSizes
        self.nEpochs = nEpochs
        self.lr = lr
        self.earlyStoppingDelta = earlyStoppingDelta
        self.earlyStoppingPatience = earlyStoppingPatience
        self.maximumNegativeLoss = maximumNegativeLoss
        self.ann = None
        self.performance = None

        return
    
    def fit(self, X, y):
        """
        """

        #
        if len(y.shape) == 1 or y.shape[0] != X.shape[0]:
            raise Exception('y must have the same size as X along the first dimension')

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self.ann = DeepNeuralNetwork(
            self.inputLayerSize,
            self.hiddenLayerSizes
        )
        nSamples = X.shape[0]
        trainIndex = np.arange(int(round(nSamples / 5 * 4)))
        testIndex = np.arange(trainIndex[-1] + 1, nSamples)
        lossFunction = nn.MSELoss()
        optimizer = optim.Adam(self.ann.parameters(), lr=self.lr)
        counterSubthresholdDelta = 0
        counterNegativeDelta = 0
        self.performance = np.full(self.nEpochs, np.nan)
        lossPrevious = np.nan
        lossMinimum = np.inf
        stateDict = None

        # Main training loop
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
                lossCurrent = lossObject.item()

            # Skip further evaluation for the first iteration
            if iEpoch == 0:
                lossPrevious = lossCurrent
                continue

            #
            lossDelta = lossPrevious - lossCurrent
            end = '\r' if iEpoch + 1 != self.nEpochs else '\n'
            print(f'Epoch {iEpoch + 1} out of {self.nEpochs}: loss={lossCurrent:4f} (delta={lossDelta:.9f})', end=end)

            # Save the performance
            self.performance[iEpoch] = lossCurrent

            # Keep track of the best cross-validated model
            if lossCurrent < lossMinimum:
                stateDict = self.ann.state_dict()

            #
            if self.maximumNegativeLoss is not None:

                # Worsening performance is measured with a cummulative counter
                if lossDelta < 0:
                    counterNegativeDelta += 1
                if counterNegativeDelta > self.maximumNegativeLoss:
                    print(f'Epoch {iEpoch + 1} out of {self.nEpochs}: loss={lossCurrent:4f} (delta={lossDelta:.9f})', end='\n')
                    break

            # Stop early if change in loss is slowing down or worsening (to prevent overfitting)
            if self.earlyStoppingDelta is not None:

                # Performance slowdown is allowed to reset if improvement is detected
                if lossDelta < self.earlyStoppingDelta:
                    counterSubthresholdDelta += 1
                else:
                    counterSubthresholdDelta = 0
                if counterSubthresholdDelta >= self.earlyStoppingPatienceeshold:
                    print(f'Epoch {iEpoch + 1} out of {self.nEpochs}: loss={lossCurrent:4f} (delta={lossDelta:.9f})', end='\n')
                    break

            # Update the previous loss variable for the next iteration
            lossPrevious = lossCurrent

        # Load the best cross-validated model
        self.ann.load_state_dict(stateDict)

        return self
    
    def predict(self, y):
        """
        """

        return self.ann(torch.tensor(y, dtype=torch.float32)).detach().numpy()
    
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
        'hiddenLayerSizes': [[10,],  [10, 10,],   [10, 10, 10,], [10, 10, 10, 10], [10, 10, 10, 10, 10],
                             [50,],  [50, 50,],   [50, 50, 50,], [50, 50, 50, 50], [50, 50, 50, 50, 50],
                             [100,], [100, 100,], [100, 100, 100,], [100, 100, 100, 100], [100, 100, 100, 100, 100],
                             [150,], [150, 150,], [150, 150, 150,], [150, 150, 150, 150], [150, 150, 150, 150, 150],
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