import subprocess


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
                ermse = float(lsp[7])
                emse = float(lsp[13])
                frmse = float(lsp[17])
                fmse = float(lsp[21])

    ref_ermse = 0.44241
    ref_emse = 0.36433
    ref_frmse = 639.68
    ref_fmse = 493.95

    assert abs(ermse - ref_ermse) < 1e-4
    assert abs(emse - ref_emse) < 1e-4
    assert abs(frmse - ref_frmse) < 1e-2
    assert abs(fmse - ref_fmse) < 1e-2
