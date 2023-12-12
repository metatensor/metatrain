# We first try to find a output of a prevouis training run

OUTDIR=$(find outputs -type d -mindepth 2 | head -n 1)
echo $OUTDIR

# And evaluate the model on the training dataset

metatensor-models eval --model=$OUTDIR/model_final.pt \
                       --structure=qm9_reduced_100.xyz \
                       --output=output.xyz

# The evaluation command predicts the property the model was trained against; here "U0".
# The predictions together with the structures have been written in a file named
# ``output.xyz`` in the current directory. The written file starts with the following
# lines

head -n 20 output.xyz

# All command line flags of the eval sub-command can be listed via

metatensor-models eval --help
