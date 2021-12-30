from models.fgrmf import FGRMF, FGRMFLoss
from utils.data import Resource
from utils.experiment import generate_model_id, run_experiment
from utils.training import NumSevereSideEffects, Scorer


def test_reproducibility():
    param_grid = {"embedding_l2": [0.0001], "sim_lambda": [0.00001]}
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
        model_class=FGRMF,
        loss_function=FGRMFLoss(),
        is_loss_extended=True,
        scorer=scorer,
        param_grid=param_grid,
        config=config,
        # loss_extension_kwargs
        sim_matrix=resource.load_sim_matrix(),
    )
    assert ap_dicts == [
        {
            0: 0.75,
            1: 0.09126984126984126,
            2: 0.10354010025062656,
            4: 0.16666666666666666,
            5: 0.7581908831908832,
            6: 0.5611531986531987,
            8: 0.14835164835164835,
            9: 0.1125,
        }
    ]
