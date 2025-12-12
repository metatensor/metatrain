Considerations for maintainers
==============================

Syncronizing with MACE hypers
-----------------------------

Whenever the official MACE updates his parameters (add new arguments
to the MACE class, change defaults...) we likely want to keep metatrain's
MACE in sync.

MACE's training scripts are based on `argparse`, and the
`./_dev/_gen_documentation.py` file can be executed to automatically convert
the argparser into the format for metatrain documentation. If you
execute the file, it will generate a `documentation.py` file in the
same folder, from which you can copy-paste bits into the actual
`documentation.py` file of the architecture.

If a new model hyper is added, one needs to add it to the `MODEL_HYPERS`
variable in `./_dev/_gen_documentation.py` before running the file.

In any case, this is just a helper script, and one can simply look at
MACE's argparser and transfer the hyperparameters manually to the
architecture's `documentation.py`.
