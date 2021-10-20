import os
import typing
from typing import Optional


from sklearn.gaussian_process.kernels import *
import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib import cm

# Custom Imports
import torch
import gpytorch
from sklearn.model_selection import ParameterSampler, train_test_split

# Define default training hyperparameters
DEFAULT_TRAINING_ITERS = 60
DEFAULT_LEARNING_RATE = 0.1

# Define if you want to optimize learning rate and number of epochs on a validation set
OPTIMIZE_PARAMETERS = False
N_SEARCHES_OPTIMIZATION = 2
MAX_ITERS_OPTIMIZATION = 1000
EARLY_STOPPING_PATIENCE = 5

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation

# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0


class GPModel(gpytorch.models.ExactGP):
    """
    Approximate GP Regressor implemented for 2D data. Uses inducing points (Interpolation Grid)
    to approximate kernel function.
    """
    def __init__(self,
                 x: torch.tensor,
                 y: torch.tensor,
                 likelihood: gpytorch.likelihoods.Likelihood,
                 mean_function: Optional[gpytorch.means.Mean] = None,
                 base_covariance_function: Optional[gpytorch.kernels.Kernel] = None):
        """

        Parameters
        ----------
        x:  torch.tensor
            Training data
        y:  torch.tensor
            Labels
        likelihood: gpytorch.likelihoods.Likelihood
            Likelihood for GP model
        mean_function: Optional(gpytorch.means.Mean)
            Mean function. Default is gpytorch.means.ConstantMean()
        base_covariance_function: Optional(gpytorch.kernels.Kernel)
            Base covariance function to be approximated. Default is gpytorch.kernels.RBFKernel().
        """
        super(GPModel, self).__init__(x, y, likelihood)

        if mean_function is None:
            self.mean_function = gpytorch.means.ConstantMean()
        else:
            self.mean_function = mean_function

        if base_covariance_function is None:
            base_covariance_function = gpytorch.kernels.RBFKernel()

        grid_size = gpytorch.utils.grid.choose_grid_size(x)

        self.covariance_function = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                base_covariance_function,
                grid_size=grid_size,
                num_dims=2
            )
        )

    def forward(self, x):
        mean = self.mean_function(x)
        covariance = self.covariance_function(x)
        distr = gpytorch.distributions.MultivariateNormal(mean, covariance)
        return distr


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.seed = 0
        self.rng = np.random.default_rng(seed=self.seed)

        self.model = None
        self.likelihood = None

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        self.model.eval()
        self.likelihood.eval()

        x = torch.Tensor(x)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model(x)
            gp_mean, gp_std = observed_pred.mean, observed_pred.stddev
            predictions = observed_pred.mean

        predictions = predictions.detach().numpy()

        # NOTE: Why do we do the detach only for predictions?
        gp_mean = gp_mean.detach().numpy()
        gp_std = gp_std.detach().numpy()

        return predictions, gp_mean, gp_std

    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray, optimize_parameters=OPTIMIZE_PARAMETERS):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        :optimize_parameters: If True, performs a hyperparameter search on the number of epochs and learning rate
        using validation data.
        """

        self.train_x = torch.Tensor(train_x)
        self.train_y = torch.Tensor(train_y)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPModel(self.train_x, self.train_y, self.likelihood)

        if optimize_parameters:
            parameters = self._optimize_parameters(n_searches=N_SEARCHES_OPTIMIZATION)

        else:
            parameters = {
                "training_iterations": DEFAULT_TRAINING_ITERS,
                "learning_rate": DEFAULT_LEARNING_RATE
            }

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=parameters.get("learning_rate"))

        self._train(self.model, self.likelihood, self.train_x, self.train_y,
                    parameters.get("training_iterations"), optimizer, mll)

    def _train(self, model, likelihood, x, y, iterations, optimizer, loss_function):
        self.losses = []
        model.train()
        likelihood.train()

        for i in range(iterations):
            print('.', end="")
            optimizer.zero_grad()
            output = model(x)
            loss = -loss_function(output, y)
            loss.backward()
            optimizer.step()
            self.losses.append(loss)

        print("\n")

    def _optimize_parameters(self, n_searches, plot_losses=True):
        parameter_grid = {
            "learning_rate": [0.1, 0.01],
        }
        param_list = list(ParameterSampler(parameter_grid, n_iter=n_searches, random_state=self.seed))
        train_x, val_x, train_y, val_y = train_test_split(self.train_x, self.train_y)

        early_stopping_patience = EARLY_STOPPING_PATIENCE
        best_val_loss = np.inf
        best_parameters = {}

        for i, parameters in enumerate(param_list):
            print(parameters)

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = GPModel(train_x, train_y, likelihood)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.model)
            optimizer = torch.optim.Adam(model.parameters(), lr=parameters.get("learning_rate"))

            val_losses, train_losses = [], []
            n_no_improvement = 0
            previous_val_loss = np.inf

            train_loss = self._get_val_loss(model, likelihood, train_x, train_y)
            val_loss = self._get_val_loss(model, likelihood, val_x, val_y)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            for iter in range(MAX_ITERS_OPTIMIZATION):
                print('.', end="")
                model.train()
                likelihood.train()

                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

                val_loss = self._get_val_loss(model, likelihood, val_x, val_y)

                train_losses.append(loss)
                val_losses.append(val_loss)

                if previous_val_loss < val_loss:
                    n_no_improvement += 1
                    if n_no_improvement == early_stopping_patience:
                        break
                else:
                    previous_val_loss = val_loss

            print("\n")

            if plot_losses:
                self._plot_losses(train_losses, val_losses, parameters)

            if val_loss < best_val_loss:
                best_parameters = parameters
                best_parameters["training_iterations"] = iter
                best_val_loss = val_loss

        print(best_parameters)
        return best_parameters

    def _get_val_loss(self, model, likelihood, val_x, val_y):
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = model(val_x)
            pred_val_y = observed_pred.mean.detach().numpy()

        val_y, pred_val_y = np.array(val_y), np.array(pred_val_y)
        val_loss = cost_function(y_true=val_y, y_predicted=pred_val_y)

        model.train()
        likelihood.train()

        return val_loss

    def _plot_losses(self, train_losses, val_losses, parameters):
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        try:
            learning_rate = parameters["learning_rate"]
            plt.title(f"Training History\nlearning rate: {learning_rate}")
        except:
            print(parameters)
        plt.xlabel("Steps")
        plt.ylabel("Negative MLL")
        plt.legend()
        plt.show()


def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fit_model(train_x, train_y)

    # Predict on the test features
    print('Predicting on test features')
    predicted_y = model.predict(test_x)
    print(predicted_y)

    # # Predict on the train features
    # print('Predicting on train features')
    # predicted_y = model.predict(train_x)[0]
    # print(predicted_y)
    #
    # print(f"Cost function on train data: {cost_function(y_predicted=predicted_y, y_true=train_y)}")

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='..')


if __name__ == "__main__":
    main()
