metatensor-models train --config-dir=. \
                        --config-name=parameters.yaml

# A model dumps file suffixes `.pt` are written to the output folder for evaluation. The
# architure-specific help page can also be accessed from the cli

metatensor-models train architecture=soap_bpnn --help

# This prints the default parameters of the architure. The general Hydra specific help
# can be accessed by

metatensor-models train --hydra-help
