import numpy as np
from sklearn.svm import SVC

from utils.data import Resource
from utils.experiment import generate_model_id, run_sklearn_external_test, save_results
from utils.training import NumSevereSideEffects

if __name__ == "__main__":
    param_grid = {
        "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "kernel": ["linear", "poly", "rbf"],
    }
    config = {"test_delete_ratio": 0}

    model_id = generate_model_id()

    resource = Resource("./data")
    data_train = resource.load_faers_train(threshold=3)
    data_whole = resource.load_faers_whole(threshold=3)
    severe_indices = np.arange(data_train.shape[1])[
        -NumSevereSideEffects.FAERS :  # noqa
    ]

    ap_dicts = run_sklearn_external_test(
        data_train=data_train,
        data_whole=data_whole,
        severe_indices=severe_indices,
        model=SVC(),
        param_grid=param_grid,
    )

    save_results(model_id, ap_dicts)
