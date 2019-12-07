import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from .utils import create_dropout_layer, create_nonlinearity_layer
from .utils import tau as utils_tau


class FCNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden, **kwargs):
        super(FCNet, self).__init__()

        self.n_hidden = n_hidden

        # Dropout related settings
        if 'dropout_rate' in kwargs:
            self.dropout_rate = kwargs['dropout_rate']
            self.dropout_type = kwargs['dropout_type']
        else:
            self.dropout_rate = 0
            self.dropout_type = 'identity'

        # Nonlinear layer setting
        if 'nonlinear_type' in kwargs:
            self.nonlinear_type = kwargs['nonlinear_type']
        else:
            self.nonlinear_type = 'relu'

        if 'learn_hetero' in kwargs:
            self.learn_hetero = kwargs['learn_hetero']

        # Setup layers
        # Input layer
        self.input = nn.ModuleDict({
            'linear': nn.Linear(input_dim, hidden_dim),
            'dropout': create_dropout_layer(
                self.dropout_rate, self.dropout_type),
            'nonlinear': create_nonlinearity_layer(self.nonlinear_type),
        })

        # Hidden Layer(s)
        if n_hidden > 0:
            self.hidden_layers = nn.ModuleList()

            for i in range(n_hidden):
                self.hidden_layers.append(
                    nn.ModuleDict({
                        'linear': nn.Linear(hidden_dim, hidden_dim),
                        'dropout': create_dropout_layer(
                            self.dropout_rate, self.dropout_type),
                        'nonlinear': create_nonlinearity_layer(
                            self.nonlinear_type),
                    })
                )

        # Hetero noise
        if self.learn_hetero:
            self.output_noise = nn.Linear(hidden_dim, output_dim)

        # Output
        self.output = nn.ModuleDict({
            'linear': nn.Linear(hidden_dim, output_dim),
            'dropout': create_dropout_layer(
                self.dropout_rate, self.dropout_type),
        })

    def forward(self, X):
        # Forward through the input layer
        activation = self.input['linear'](X)
        activation = self.input['dropout'](activation)
        activation = self.input['nonlinear'](activation)

        # Forward through hidden layers
        if hasattr(self, 'hidden_layers'):
            for hidden in self.hidden_layers:
                activation = hidden['linear'](activation)
                activation = hidden['dropout'](activation)
                activation = hidden['nonlinear'](activation)

        if self.learn_hetero:
            noise = self.output_noise(activation)
        else:
            noise = None

        activation = self.output['linear'](activation)
        activation = self.output['dropout'](activation)

        return activation, noise

    def predict_dist(self, test_data, test_data_have_targets=True, tau=None,
                     n_prediction=1000, **kwargs):

        # Check whether self (network) was in training mode or testing mode
        was_eval = not self.training

        # Target detransform (denormalization)
        if 'y_mean' in kwargs:
            y_mean = kwargs['y_mean']
            y_std = kwargs['y_std']
        else:
            y_mean = 0
            y_std = 1

        # Parameters for Tau calculation
        if not self.learn_hetero:
            if tau is None:
                raise Exception(
                    "tau needs to be specified for homoscedastic noise.")

        metrics = {}

        if isinstance(test_data, torch.utils.data.DataLoader):

            predictions = []
            mean = 0

            # if test_data includes targets or y_test is given,
            # We prepare variables for evaluation metrics
            if test_data_have_targets or ('y_test' in kwargs):

                metrics['rmse_mc'] = 0
                metrics['rmse_non_mc'] = 0
                metrics['test_ll_mc'] = 0

            if 'y_test' in kwargs:
                y_test = kwargs['y_test']
            else:
                y_test = [None for _ in len(test_data)]

            # We will assume that y will be prepared to have
            # same number of data points as
            # data from test_data
            for data, y in zip(test_data, y_test):
                if test_data_have_targets:
                    inputs, targets = data
                else:
                    inputs = data
                    targets = None

                if y is not None:
                    assert len(inputs) == len(y)
                    targets = y

                # Denormalization
                if targets is not None:
                    targets = targets * y_std + y_mean

                # Determine where our test data needs to be sent to
                # by checking the first fc layer weight's location
                first_weight_location = self.input['linear'].weight.device

                inputs = inputs.to(first_weight_location)

                # Explictly send targets to device memory only when
                # it is coming from test_data DataLoader
                if test_data_have_targets:
                    targets = targets.to(first_weight_location)

                # Temporaily disable eval mode
                if was_eval:
                    self.train()

                predictions_batch = torch.stack(
                    [self.forward(inputs) for _ in range(n_prediction)])

                if was_eval:
                    self.eval()

                mean_batch = torch.mean(predictions_batch, 0)

                mean += mean_batch
                mean /= 2

                predictions.append(predictions_batch)

                if len(metrics) > 0:
                    # RMSE
                    metrics['rmse_mc'] += torch.mean(
                        torch.pow(target - mean_batch, 2))
                    metrics['rmse_mc'] /= 2

                    # RMSE (Non-MC)
                    prediction_non_mc = self.forward(X_test)
                    prediction_non_mc = prediction_non_mc * y_std + y_mean

                    metrics['rmse_non_mc'] += torch.mean(
                        torch.pow(target - prediction_non_mc, 2))
                    metrics['rmse_non_mc'] /= 2

                    # test log-likelihood
                    metrics['test_ll_mc'] -= torch.mean(
                        torch.logsumexp(
                            - torch.tensor(0.5) * tau * torch.pow(
                                y_test[None] - predictions, 2), 0)
                        - torch.log(
                            torch.tensor(n_predictions, dtype=torch.float))
                        - torch.tensor(0.5) * torch.log(
                            torch.tensor(2 * np.pi, dtype=torch.float))
                        + torch.tensor(0.5) * torch.log(tau)
                    )
                    metrics['test_ll_mc'] /= 2

            predictions = torch.cat(predictions)
            var = torch.var(predictions)

            if len(metrics) > 0:
                metrics['rmse_mc'] = torch.sqrt(metrics['rmse_mc'])
                metrics['rmse_non_mc'] = torch.sqrt(metrics['rmse_non_mc'])

        # Assuming test_data is given in non-iterable format
        else:
            # Temporaily disable eval mode
            if was_eval:
                self.train()

            predictions = []
            noises = []

            for _ in range(n_prediction):
                outputs, noise = self.forward(test_data)
                predictions.append(outputs)

                if noise is not None:
                    noises.append(noise.exp())

            predictions = torch.stack(predictions)

            predictions = predictions * y_std + y_mean

            if was_eval:
                self.eval()

            mean = torch.mean(predictions, 0)

            # Epistemic variance
            var = torch.var(predictions, 0)

            # If noises were learned
            if len(noises) > 0:
                noises = torch.stack(noises)
                noises = torch.mean(torch.pow(noises, 2), 0)
            else:
                # homoscedastic noise
                noises = (1/tau) * torch.ones_like(var)

            # If y_test is given, calculate RMSE and test log-likelihood
            if 'y_test' in kwargs:
                y_test = kwargs['y_test']
                y_test = y_test * y_std + y_mean

                # RMSE
                metrics['rmse_mc'] = torch.sqrt(
                    torch.mean(torch.pow(y_test - mean, 2)))

                # RMSE (Non-MC)
                prediction_non_mc, _ = self.forward(test_data)
                prediction_non_mc = prediction_non_mc * y_std + y_mean
                metrics['rmse_non_mc'] = torch.sqrt(
                    torch.mean(torch.pow(y_test - prediction_non_mc, 2)))

                # test log-likelihood
                metrics['test_ll_mc'] = torch.mean(
                    torch.logsumexp(
                        - torch.tensor(0.5) * tau
                        * torch.pow(y_test[None] - predictions, 2), 0)
                    - torch.log(
                        torch.tensor(n_prediction, dtype=torch.float))
                    - torch.tensor(0.5) * torch.log(
                        torch.tensor(2 * np.pi, dtype=torch.float))
                    + torch.tensor(0.5) * torch.log(tau)
                )

        return predictions, mean, var, noises, metrics
