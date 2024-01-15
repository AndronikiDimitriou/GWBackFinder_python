#%%
import pickle
import numpy as np
from scipy import stats
import torch
import sbi
from sbi.utils import process_prior
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
import pathlib
basepath = pathlib.Path(__file__).parent.resolve()
# %%
    
def train(thetas=None, gw_total=None, prior=None, resume_training=False, validation_fraction=0.2, 
          learning_rate=1e-4, show_train_summary=True, max_num_epochs=None, path_saved=None, path_inference=None,
          model_type="nsf", hidden_features=64, num_transforms=3, name_file=None):
    
    """
    Train a density estimator using Sequential Neural Likelihood (SNPE) algorithm.

    Parameters:
    
    - thetas: Parameters for the simulations.
    - gw_total: Observed data.
    - prior: Prior distribution.
    - resume_training: If True, resume training from a saved model.
    - validation_fraction: Fraction of the data used for validation during training.
    - learning_rate: Learning rate for the optimizer during training.
    - show_train_summary: If True, show training summary.
    - max_num_epochs: Maximum number of epochs for training.
    - path_saved: Path to the saved model if resume_training is True.
    - path_inference: Path to save inference.
    - model_type: Type of neural network model.
    - hidden_features: Number of hidden features in the neural network model.
    - num_transforms: Number of transformations in the neural network model.
    - name_file: Name of the pickle file (without extension) to save the inference object.
    Returns:
    Trained density estimator.
    """

    if resume_training==True:
        with open(path_saved, "rb") as handle:
            inference = pickle.load(handle)

        density_estimator = inference.train(resume_training=True, validation_fraction=validation_fraction, 
                                            learning_rate=learning_rate, show_train_summary=show_train_summary, 
                                            max_num_epochs=max_num_epochs, force_first_round_loss=True)

        with open(path_inference+name_file, "wb") as handle:
            pickle.dump(inference, handle) 
    
    else: 
        density_estimator_build_fun = posterior_nn(model=model_type, hidden_features=hidden_features, 
                                                    num_transforms=num_transforms)
        inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)

        inference = inference.append_simulations(thetas, gw_total)

        density_estimator = inference.train(resume_training=False, validation_fraction=validation_fraction, 
                                            learning_rate=learning_rate, show_train_summary=show_train_summary, 
                                            max_num_epochs=max_num_epochs, force_first_round_loss=False)
        
        with open(path_inference+name_file, "wb") as handle:
            pickle.dump(inference, handle) 
    
    return density_estimator    

def get_posterior(path,name_file):
    """
    Load a trained inference object from a pickle file, build the posterior,
    and save the posterior to another pickle file.

    Parameters:
    - path: Path to the saved inference object pickle file.
    - name_file: Name of the pickle file (without extension) to save the posterior.

    Returns:
    - posterior: Built posterior object.
    """
    with open(path, "rb") as handle:
        inference = pickle.load(handle)
    posterior = inference.build_posterior()

    with open("./"+name_file, "wb") as handle:
        pickle.dump(posterior, handle)
    return posterior
# %%