from models.mf import MF, MFLoss
from utils.data import Resource
from utils.experiment import generate_model_id, run_external_test
from utils.training import NumSevereSideEffects, Scorer


def test_reproducibility():
    param_grid = {"embedding_l2": [0.0001]}
    config = {
        "k": 100,
        "epochs": 1,
        "lr": 0.01,
        "optimizer_name": "adam",
        "scheduler_name": "onplateau",
        "es_patience": 2,
        "sc_patience": 0,
        "test_delete_ratio": 0,
    }

    resource = Resource("./data")
    data_train = resource.load_faers_train(threshold=3)[:100, :10]
    data_whole = resource.load_faers_whole(threshold=3)[:100, :10]
    scorer = Scorer(data_train.shape[1], NumSevereSideEffects.FAERS)

    ap_dicts = run_external_test(
        model_id=generate_model_id(),
        data_train=data_train,
        data_whole=data_whole,
        model_class=MF,
        loss_function=MFLoss(),
        is_loss_extended=False,
        scorer=scorer,
        param_grid=param_grid,
        config=config,
    )
    assert ap_dicts == [
        {
            0: 0.6309523809523809,
            1: 0.04945054945054945,
            2: 0.03538461538461539,
            5: 0.15050052119017637,
            6: 0.3279030910609858,
            8: 0.04470046082949309,
            9: 0.059947299077733864,
        }
    ]
