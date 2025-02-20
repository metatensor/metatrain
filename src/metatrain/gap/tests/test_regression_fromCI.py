import subprocess

import torch


torch.set_default_dtype(torch.float64)  # GAP only supports float64


def test_regression_CI():
    """Perform a regression test on the model at initialization"""

    # Run the first command and redirect output to a file
    with open("out.put", "w") as output_file:
        subprocess.run(
            ["mtt", "train", "options-gap.yaml"],
            stdout=output_file,
            stderr=subprocess.STDOUT,
        )

    with open("out.put", "r") as fil:
        lines = fil.readlines()
        for line in lines:
            lsp = line.split()
            if len(lsp) > 5 and lsp[4] == "RMSE":
                ermse = lsp[7]
                emse = lsp[13]
                frmse = lsp[17]
                fmse = lsp[21]

    ref_ermse = str(0.44241)
    ref_emse = str(0.36433)
    ref_frmse = str(639.68)
    ref_fmse = str(493.95)

    assert ermse == ref_ermse
    assert emse == ref_emse
    assert frmse == ref_frmse
    assert fmse == ref_fmse
