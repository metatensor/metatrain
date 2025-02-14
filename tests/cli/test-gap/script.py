with open("out.put", "r") as fil:
    lines = fil.readlines()
    for line in lines:
        lsp = line.split()
        if len(lsp) > 5 and lsp[4] == "RMSE":
            ermse = lsp[7]
            emse = lsp[13]
            frmse = lsp[17]
            fmse = lsp[21]
with open("reference.out", "r") as fil:
    lines = fil.readlines()
    lsp = lines[2].split()
    ref_ermse = lsp[0]
    ref_emse = lsp[1]
    ref_frmse = lsp[2]
    ref_fmse = lsp[3]

print("computed values")
print(ermse, emse, frmse, fmse)
print("")
print("reference values")
print(ref_ermse, ref_emse, ref_frmse, ref_fmse)
print("")


assert ermse == ref_ermse
assert emse == ref_emse
assert frmse == ref_frmse
assert fmse == ref_fmse

print("All test passed!!!")
