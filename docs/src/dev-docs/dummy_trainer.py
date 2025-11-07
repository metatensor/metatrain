from metatrain.utils.abc import TrainerInterface


# Definition of hyperparameters and their defaults:
class TrainerHypers:
    learning_rate = 1e-3
    lr_scheduler = "CosineAnnealing"
    ...


class MyTrainer(TrainerInterface):
    __checkpoint_version__ = 1
    __hypers_cls__ = TrainerHypers

    def __init__(self, hypers: dict):
        super().__init__(hypers)
        # To access hyperparameters, one can use self.hypers, which
        # by default will be {'learning_rate': 1e-3, 'lr_scheduler': 'CosineAnnealing'}
        self.hypers["learning_rate"]
        ...

    # Here one would implement the rest of the abstract methods
