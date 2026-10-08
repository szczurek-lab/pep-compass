import json
from pathlib import Path

import torch

from pep_compass.geodesic_search.train_properties_aproximating_model import (
    Distribution_Based_Regression_Transformer,
)


def load_predictor_models(mean_model_path = "Best_mean_predictor.pth",
                           mean_config_path = "Best_mean_config.json",
                             var_model_path = "Best_var_predictor.pth",
                               var_config_path = "Best_var_config.json",
                               Trained_predictor_models_path: Path | None = None,
                               device = 'cpu'):
    """
    Loads two trained predictor models (mean and variance estimators) from specified file paths.
    Best to pass Trained_predictor_models_path as a path to the Trained_predictor_models folder. Then
    models will be corectly loaded

    Args:
        mean_model_path (str, optional): Path to the mean predictor model checkpoint (.pth). 
            Defaults to "Best_mean_predictor.pth".
        mean_config_path (str, optional): Path to the JSON configuration file for the mean predictor model. 
            Defaults to "Best_mean_config.json".
        var_model_path (str, optional): Path to the variance predictor model checkpoint (.pth). 
            Defaults to "Best_var_predictor.pth".
        var_config_path (str, optional): Path to the JSON configuration file for the variance predictor model. 
            Defaults to "Best_var_config.json".
        Trained_predictor_models_path (Path | None, optional): If specified, prepends this directory path 
            to all model and configuration file paths. Defaults to None.

    Returns:
        Tuple[torch.nn.Module, torch.nn.Module]: The loaded mean and variance predictor models in evaluation mode.
    """

    
    if Trained_predictor_models_path != None:
        mean_model_path = Trained_predictor_models_path / mean_model_path
        var_model_path = Trained_predictor_models_path / var_model_path
        mean_config_path = Trained_predictor_models_path / mean_config_path
        var_config_path = Trained_predictor_models_path / var_config_path
    
    with open(mean_config_path, "r") as f:
        args = json.load(f)
    mean_model = Distribution_Based_Regression_Transformer(
        args["model_dim"], args["num_heads"], args["num_layers"], args["hidden_ff_dim"],
        args["dropout"], args["number_of_predictions"], seq_len=25, alphabet_size=21,
        device = device
    )
    mean_model.load_state_dict(torch.load(mean_model_path, map_location = device))

    with open(var_config_path, "r") as f:
        args = json.load(f)
    var_model = Distribution_Based_Regression_Transformer(
        args["model_dim"], args["num_heads"], args["num_layers"], args["hidden_ff_dim"],
        args["dropout"], args["number_of_predictions"], seq_len=25, alphabet_size=21,
        device = device
    )
    var_model.load_state_dict(torch.load(var_model_path, map_location = device))
    var_model.eval()
    mean_model.eval()

    return mean_model, var_model

def load_MIC_predictor_model(model_path = "MICs_predictor_model.pth",
                            config_path = "MICs_predictor_config.json",
                            Trained_predictor_models_path: Path | None = None,
                            device = 'cpu'):
    """
    Loads a trained MIC predictor model from the specified file paths.

    Args:
        model_path (str, optional): Path to the MIC predictor model checkpoint (.pth).
            Defaults to "MICs_predictor_model.pth".
        config_path (str, optional): Path to the JSON configuration file for the MIC predictor model.
            Defaults to "MICs_predictor_config.json".
        Trained_predictor_models_path (Path | None, optional): If specified, prepends this directory path
            to the model and configuration file paths. Defaults to None.

    Returns:
        torch.nn.Module: The loaded MIC predictor model in evaluation mode.
    """
    model, _ = load_predictor_models(model_path, config_path, Trained_predictor_models_path=Trained_predictor_models_path, device=device)
    model.eval()
    return model