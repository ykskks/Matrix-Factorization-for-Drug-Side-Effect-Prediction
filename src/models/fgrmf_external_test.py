from models.fgrmf import FGRMF, FGRMFLoss
from utils.data import Resource
from utils.experiment import generate_model_id, run_external_test, save_results
from utils.training import NumSevereSideEffects, Scorer

if __name__ == "__main__":
    param_grid = {
        "embedding_l2": [0.0001, 0.0005, 0.001, 0.005, 0.01],
        "sim_lambda": [0.00001, 0.00005, 0.0001, 0.0005, 0.001],
    }
    config = {
        "k": 100,
        "epochs": 100,
        "lr": 0.01,
        "optimizer_name": "adam",
        "scheduler_name": "onplateau",
        "es_patience": 2,
        "sc_patience": 0,
        "test_delete_ratio": 0,
    }

    model_id = generate_model_id()

    resource = Resource("./data")
    data_train = resource.load_faers_train(threshold=3)
    data_whole = resource.load_faers_whole(threshold=3)
    scorer = Scorer(data_train.shape[1], NumSevereSideEffects.FAERS)

    ap_dicts = run_external_test(
        model_id=model_id,
        data_train=data_train,
        data_whole=data_whole,
        model_class=FGRMF,
        loss_function=FGRMFLoss(),
        is_loss_extended=False,
        scorer=scorer,
        param_grid=param_grid,
        config=config,
        # loss_ingredients_kwargs
        sim_matrix=resource.load_sim_matrix(),
    )

    save_results(model_id, ap_dicts)
