#!\bin\bash

metatensor-models train options.yaml

# The above script can be found in the `scripts` folder of the repository.
python ../../scripts/setup.py

metatensor-models export model.pt -o exported_model_ethanol.pt
