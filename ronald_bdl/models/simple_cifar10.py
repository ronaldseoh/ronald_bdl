import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import create_dropout_layer, create_nonlinearity_layer_functional


class SimpleCIFAR10(nn.Module):
    def __init__(self, **kwargs):
        super(SimpleCIFAR10, self).__init__()

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

        self.nonlinear_function = \
            create_nonlinearity_layer_functional(self.nonlinear_type)

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.conv1_dropout = create_dropout_layer(
            self.dropout_rate, self.dropout_type)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.conv2_dropout = create_dropout_layer(
            self.dropout_rate, self.dropout_type)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.fc1_dropout = create_dropout_layer(
            self.dropout_rate, self.dropout_type)

        self.fc2 = nn.Linear(120, 84)

        self.fc2_dropout = create_dropout_layer(
            self.dropout_rate, self.dropout_type)

        self.fc3 = nn.Linear(84, 10)

        self.fc3_dropout = create_dropout_layer(
            self.dropout_rate, self.dropout_type)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(
            self.nonlinear_function(self.conv1_dropout(self.conv1(x))))
        x = self.pool(
            self.nonlinear_function(self.conv2_dropout(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = self.nonlinear_function(self.fc1_dropout(self.fc1(x)))
        x = self.nonlinear_function(self.fc2_dropout(self.fc2(x)))
        x = self.fc3_dropout(self.fc3(x))

        return x

    def predict_dist(self, test_data, test_data_have_targets=True,
                     n_prediction=1000, **kwargs):

        was_eval = not self.training

        metrics = {}

        if isinstance(test_data, torch.utils.data.DataLoader):

            predictions = []
            mean_predictions = []

            # if test_data includes targets or y_test is given,
            # We prepare variables for evaluation metrics
            if test_data_have_targets or ('y_test' in kwargs):

                metrics['accuracy_mc'] = 0
                metrics['accuracy_non_mc'] = 0
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

                # Determine where our test data needs to be sent to
                # by checking the first conv layer weight's location
                first_weight_location = self.conv1.weight.device

                inputs = inputs.to(first_weight_location)

                # Explictly send targets to device memory only when
                # it is coming from test_data DataLoader
                if test_data_have_targets:
                    targets = targets.to(first_weight_location)

                # Temporaily disable eval mode
                if was_eval:
                    self.train()

                raw_scores_batch = torch.stack(
                    [self.forward(inputs) for _ in range(n_prediction)])

                if was_eval:
                    self.eval()

                predictions_batch = torch.max(raw_scores_batch, 2).values
                predictions.append(predictions_batch)

                mean_raw_scores_batch = torch.mean(raw_scores_batch, 0)
                mean_predictions_batch = torch.argmax(
                    mean_raw_scores_batch, 1)
                mean_predictions.append(mean_predictions_batch)

                non_mc_raw_scores_batch = self.forward(inputs)
                non_mc_predictions_batch = torch.argmax(
                    non_mc_raw_scores_batch, 1)

                if len(metrics) > 0:
                    # Accuracy
                    metrics['accuracy_mc'] += torch.mean(
                        (mean_predictions_batch == targets).float())
                    metrics['accuracy_mc'] /= 2

                    # Accuracy (Non-MC)
                    metrics['accuracy_non_mc'] += torch.mean(
                        (non_mc_predictions_batch == targets).float())
                    metrics['accuracy_non_mc'] /= 2

                    # test log-likelihood
                    metrics['test_ll_mc'] -= (
                        F.cross_entropy(mean_raw_scores_batch, targets))
                    metrics['test_ll_mc'] /= 2

            mean_predictions = torch.cat(mean_predictions)
            predictions = torch.cat(predictions)

        else:
            raise Exception(
                'SimpleCIFAR10 predict_dist() '
                'only accepts DataLoader test data at the moment.')

        return predictions, mean_predictions, metrics
