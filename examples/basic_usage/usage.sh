#!\bin\bash

metatensor-models train options.yaml

# The functions saves the final model `model.pt` to the current output folder for later
# evaluation. All command line flags of the train sub-command can be listed via

metatensor-models train --help

# We now evaluate the model on the training dataset, where the first arguments specifies
# trained model and the second an option file containing the path of the dataset for evaulation.

metatensor-models eval model.pt eval.yaml

# The evaluation command predicts those properties the model was trained against; here
# "U0". The predictions together with the structures have been written in a file named
# ``output.xyz`` in the current directory. The written file starts with the following
# lines

head -n 20 output.xyz

# All command line flags of the eval sub-command can be listed via

metatensor-models eval --help

# However, before we export the model, we need to run the following command to
# hotfix errors in metatensor.

python ../../scripts/hotfix_metatensor.py

# Finally, the `metatestor-models export`, i.e.,

metatensor-models export model.pt

# creates an `exported-model.pt` file that contains the exported model.
