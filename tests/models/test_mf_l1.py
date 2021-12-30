from models.mf_l1 import MFL1, MFL1Loss
from utils.data import Resource
from utils.experiment import generate_model_id, run_experiment
from utils.training import NumSevereSideEffects, Scorer


def test_reproducibility():
    param_grid = {"embedding_l1": [0.00001]}
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
    data = resource.load_faers_train(threshold=3)[:100, :10]
    scorer = Scorer(data.shape[1], NumSevereSideEffects.FAERS)

    ap_dicts = run_experiment(
        model_id=generate_model_id(),
        data=data,
        model_class=MFL1,
        loss_function=MFL1Loss(),
        is_loss_extended=True,
        scorer=scorer,
        param_grid=param_grid,
        config=config,
    )
    assert ap_dicts == [
        {
            0: 0.7658730158730158,
            1: 0.08680555555555555,
            2: 0.10435178856231488,
            4: 0.16666666666666666,
            5: 0.7523148148148149,
            6: 0.5438692480359146,
            8: 0.13942307692307693,
            9: 0.12222222222222222,
        }
    ]
