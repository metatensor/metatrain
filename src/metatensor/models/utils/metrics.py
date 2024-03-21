from typing import Dict, Tuple

from metatensor.torch import TensorMap


class RMSEAccumulator:
    def __init__(self):
        self.information: Dict[str, Tuple[float, int]] = {}

    def update(self, predictions: Dict[str, TensorMap], targets: Dict[str, TensorMap]):
        for key, target in targets.items():
            if key not in self.information:
                self.information[key] = (0.0, 0)
            prediction = predictions[key]

            self.information[key] = (
                self.information[key][0]
                + ((prediction.block().values - target.block().values) ** 2)
                .sum()
                .item(),
                self.information[key][1] + prediction.block().values.numel(),
            )

            for gradient_name, target_gradient in target.block().gradients():
                if f"{target}_{gradient_name}_gradients" not in self.information:
                    self.information[f"{key}_{gradient_name}_gradients"] = (0.0, 0)
                prediction_gradient = prediction.block().gradient(gradient_name)
                self.information[f"{key}_{gradient_name}_gradients"] = (
                    self.information[f"{key}_{gradient_name}_gradients"][0]
                    + ((prediction_gradient.values - target_gradient.values) ** 2)
                    .sum()
                    .item(),
                    self.information[f"{key}_{gradient_name}_gradients"][1]
                    + prediction_gradient.values.numel(),
                )

    def finalize(self) -> Dict[str, float]:
        finalized_info = {}
        for key, value in self.information.items():
            finalized_info[f"{key} RMSE"] = (value[0] / value[1]) ** 0.5

        return finalized_info
