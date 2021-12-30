import numpy as np
from sklearn.svm import SVC

from utils.data import Resource
from utils.experiment import run_sklearn_external_test
from utils.training import NumSevereSideEffects


def test_reproducibility():
    param_grid = {"C": [0.00001], "kernel": ["linear"], "probability": [True]}

    resource = Resource("./data")
    data_train = resource.load_faers_train(threshold=3)[:100, :10]
    data_whole = resource.load_faers_whole(threshold=3)[:100, :10]
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

    assert ap_dicts == [
        {
            0: 0.35675381263616557,
            1: 0.15357142857142858,
            2: 0.0992063492063492,
            5: 0.18282697694462396,
            6: 0.08943228454172367,
            8: 0.142512077294686,
        }
    ]
