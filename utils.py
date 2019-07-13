import torch

FILE_VERSION=1

class NoModelFoundException(Exception):
    pass

def save_model_scaler(model, scaler, path):
    """
    Helper function to save a model+scaler to a file.  Supports one model/scaler pair per # of d spacings.  Updates file if it already exists.  Throws an exception if path does not exist.
    :param model: Torch trained model object to be saved
    :param scaler: scikit-learn scaler to be saved
    :param path: Path to be saved to
    :return: None
    """
    model_key = 'model_' + str(model.num_spacings)
    scaler_key = 'scaler_' + str(model.num_spacings)
    try:
        saved_model = torch.load(path)
        saved_model[model_key] = model.state_dict()
        saved_model[scaler_key] = scaler
        saved_model['version'] = FILE_VERSION
        torch.save(saved_model, path)


    except FileNotFoundError:
        torch.save({model_key: model.state_dict(), scaler_key: scaler, 'version': FILE_VERSION}, path)

def load_model_scaler(path, num_spacings):
    """
    Loads a model and scaler from a path, for a given number of d-spacings.  Throws a FileNotFoundError if path doesn't exist and NoModelFoundException if model with desired d-spacings is not present.
    :param path: path to find the saved model
    :param num_spacings: number of d-spacings the model should be trained on
    :return: dictionary {'model': model, 'scaler': scaler} of the saved model
    """
    model_key = 'model_' + str(num_spacings)
    scaler_key = 'scaler_' + str(num_spacings)
    saved_model = torch.load(path) #throws FileNotFoundError if not found, caught later
    if model_key in saved_model and scaler_key in saved_model:
        return {'model': saved_model[model_key], 'scaler': saved_model[scaler_key]}
    else:
        raise NoModelFoundException
