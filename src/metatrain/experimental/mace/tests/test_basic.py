from metatrain.utils.testing import (
    ArchitectureTests,
    AutogradTests,
    CheckpointTests,
    ExportedTests,
    InputTests,
    OutputTests,
    TorchscriptTests,
    TrainingTests,
)


class MACETests(ArchitectureTests):
    architecture = "experimental.mace"


class TestInput(InputTests, MACETests): ...


class TestOutput(OutputTests, MACETests): ...


class TestAutograd(AutogradTests, MACETests): ...


class TestTorchscript(TorchscriptTests, MACETests):
    float_hypers = ["cutoff"]


class TestExported(ExportedTests, MACETests): ...


class TestTraining(TrainingTests, MACETests): ...


class TestCheckpoints(CheckpointTests, MACETests):
    incompatible_trainer_checkpoints = []
