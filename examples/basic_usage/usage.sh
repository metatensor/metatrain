#!\bin\bash

mtt train options.yaml

# The functions saves the final model `model.pt` to the current output folder for later
# evaluation. An `extensions/` folder, which contains the compiled extensions for the model,
# might also be saved depending on the architecture.
# All command line flags of the train sub-command can be listed via

mtt train --help

# We now evaluate the model on the training dataset, where the first arguments specifies
# trained model and the second an option file containing the path of the dataset for evaulation.
# The extensions of the model, if any, can be specified via the `-e` flag.

mtt eval model.pt eval.yaml -e extensions/

# The evaluation command predicts those properties the model was trained against; here
# "U0". The predictions together with the systems have been written in a file named
# ``output.xyz`` in the current directory. The written file starts with the following
# lines

head -n 20 output.xyz

# All command line flags of the eval sub-command can be listed via

mtt eval --help
