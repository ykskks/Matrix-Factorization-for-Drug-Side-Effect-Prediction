import numpy as np
from sklearn.svm import SVC

from utils.data import Resource
from utils.experiment import run_sklearn_experiment
from utils.training import NumSevereSideEffects


def test_reproducibility():
    param_grid = {"C": [0.00001], "kernel": ["linear"], "probability": [True]}
    config = {"test_delete_ratio": 0}

    resource = Resource("./data")
    data = resource.load_faers_train(threshold=3)[:100, :10]
    severe_indices = np.arange(data.shape[1])[-NumSevereSideEffects.FAERS :]  # noqa

    ap_dicts = run_sklearn_experiment(
        data=data,
        severe_indices=severe_indices,
        model=SVC(),
        param_grid=param_grid,
        config=config,
    )

    assert ap_dicts == [
        {
            0: 0.5488612836438923,
            5: 0.7475633528265108,
            6: 0.8541666666666667,
            8: 0.29166666666666663,
        }
    ]
