The files in the `templates` folder are (possibly) processed and then a directory
called `generated/` is created with the final .rst files of each architecture. To
find the template for an architecture, we first look for a file called 
`<architecture>.rst`, and if it is not there we take `generic.rst`. 

This is done in `docs/conf.py`, in a function called `setup_architecture_docs()`.
Note that you can create an rst file under templates that is completely static,
i.e. not a template at all. In that case the effect of the function will be to
just copy the file to the `generated/` directory.