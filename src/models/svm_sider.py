import numpy as np
from sklearn.svm import SVC

from utils.data import Resource
from utils.experiment import generate_model_id, run_sklearn_experiment, save_results
from utils.training import NumSevereSideEffects

if __name__ == "__main__":
    param_grid = {
        "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        "kernel": ["linear", "poly", "rbf"],
    }
    config = {"test_delete_ratio": 0}

    model_id = generate_model_id()

    resource = Resource("./data")
    data = resource.load_sider()
    severe_indices = np.arange(data.shape[1])[-NumSevereSideEffects.SIDER :]  # noqa

    ap_dicts = run_sklearn_experiment(
        data=data,
        severe_indices=severe_indices,
        model=SVC(),
        param_grid=param_grid,
        config=config,
    )
    save_results(model_id, ap_dicts)
