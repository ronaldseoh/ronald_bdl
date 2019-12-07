import torch
from torch.utils.data import Dataset


class ToyDatasets(Dataset):

    def __init__(self, random_seed=691, n_samples=20,
                 x_low=-4, x_high=4, y_noise_mean=0, y_noise_std=9,
                 random_x=True,
                 more_noise_x_from=None,
                 more_noise_x_to=None,
                 more_noise_y_from=None,
                 more_noise_y_to=None,
                 transform=None):

        self._generator = torch.Generator()
        self._generator.manual_seed(random_seed)

        self.transform = transform

        if random_x:
            # Toy dataset generation based on
            # Hernandez-Lobato & Adams, 2015 (https://arxiv.org/abs/1502.05336)
            # Also refer to
            # https://github.com/pawni/BayesByHypernet/blob/master/toy_data.ipynb
            # https://pytorch.org/docs/master/tensors.html#torch.Tensor.uniform_
            # https://pytorch.org/docs/master/tensors.html#torch.Tensor.normal_
            self.data_x = torch.empty(n_samples, 1).uniform_(
                x_low, x_high, generator=self._generator)
        else:
            # Create evenly spaced n_sample number of samples
            # between x_low and x_high
            self.data_x = torch.linspace(x_low, x_high, n_samples)
            self.data_x = self.data_x.reshape(n_samples, -1)

        # Construct self.data_y with function values and noise
        self.data_y_function = torch.pow(self.data_x, 3)
        self.data_y_noise = torch.empty(n_samples, 1).normal_(
            mean=y_noise_mean, std=y_noise_std, generator=self._generator)

        if more_noise_x_from is not None:
            applicable_x = (self.data_x >= more_noise_x_from)\
                         * (self.data_x <= more_noise_x_to)

            additional_noise = torch.empty_like(
                self.data_y_noise[applicable_x])

            # add uniform noise
            additional_noise = additional_noise.uniform_(
                more_noise_y_from, more_noise_y_to,
                generator=self._generator)

            self.data_y_noise[applicable_x] \
                += additional_noise

        self.data_y = self.data_y_function + self.data_y_noise

        self.n_features = self.data_x.shape[1]
        self.n_targets = self.data_y.shape[1]

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(idx) != list:
            idx_list = [idx]
        else:
            idx_list = idx

        sample = (self.data_x[idx_list], self.data_y[idx_list])

        return sample
