.. _contributing:

Contributing
============
🎉 First off, thanks for taking the time to contribute to metatrain! 🎉

If you want to contribute but feel a bit lost, do not hesitate to contact us and ask
your questions! We will happily mentor you through your first contributions.

Area of contributions
---------------------
The first and best way to contribute to metatrain is to use it and advertise it
to other potential users. Other than that, you can help with:

- documentation: correcting typos, making various documentation clearer;
- bug fixes and improvements to existing code;
- adding new architectures
- and many more ...

All these contributions are very welcome. We accept contributions via Github `pull
request <https://github.com/metatrain/pulls>`_. If you want to work on the code
and pick something easy to get started, have a look at the `good first issues
<https://github.com/metatensor/metatrain/labels/Good%20first%20issue>`_.


Bug reports and feature requests
--------------------------------
Bug and feature requests should be reported as `Github issues
<https://github.com/metatrain/issues>`_. For bugs, you should provide
information so that we can reproduce it: what did you try? What did you expect? What
happened instead? Please provide any useful code snippet or input file with your bug
report.

If you want to add a new feature to metatrain, please create an `issue
<https://github.com/metatensor/metatrain/issues/new>`_ so that we can discuss it,
and you have more chances to see your changes incorporated.


Contribution tutorial
---------------------
In this small tutorial, you should replace `<angle brackets>` as needed. If anything is
unclear, please ask for clarifications! There are no dumb questions.

Getting started
---------------
To help with developing start by installing the development dependencies:

.. code-block:: bash

  pip install tox


Then this package itself

.. code-block:: bash

  git clone https://github.com/metatensor/metatrain 
  cd metatrain 
  pip install -e .

This install the package in development mode, making it importable globally and allowing
you to edit the code and directly use the updated version. To see a list of all
supported tox environments please use

.. code-block:: bash

  tox list

Running the tests
-----------------
The testsuite is implemented using Python's `unittest`_ framework and should be set-up
and run in an isolated virtual environment with `tox`_. All tests can be run with

.. code-block:: bash

  tox                  # all tests

If you wish to test only specific functionalities, for example:

.. code-block:: bash

  tox -e lint          # code style
  tox -e tests         # unit tests of the main library
  tox -e examples      # test the examples


You can also use ``tox -e format`` to use tox to do actual formatting instead of just
testing it. Also, you may want to setup your editor to automatically apply the `black
<https://black.readthedocs.io/en/stable/>`_ code formatter when saving your files, there
are plugins to do this with `all major editors
<https://black.readthedocs.io/en/stable/editor_integration.html>`_.

If you want to test a specific archicture you can also do it. For example

.. code-block:: bash

      tox -e soap-bpnn-tests

Will run the unit and regression tests for the :ref:`SOAP-BPNN <architecture-soap-bpnn>`
model. Note that architecture tests are not run by default if you just type ``tox``.

.. _unittest: https://docs.python.org/3/library/unittest.html
.. _tox: https://tox.readthedocs.io/en/latest

Contributing to the documentation
---------------------------------
The documentation is written in reStructuredText (rst) and uses `sphinx`_ documentation
generator. In order to modify the documentation, first create a local version on your
machine as described above. Then, build the documentation with

.. code-block:: bash

    tox -e docs

You can then visualize the local documentation with your favorite browser using the
following command (or open the :file:`docs/build/html/index.html` file manually).

.. code-block:: bash

    # on linux, depending on what package you have installed:
    xdg-open docs/build/html/index.html
    firefox docs/build/html/index.html

    # on macOS:
    open docs/build/html/index.html

.. _`sphinx` : https://www.sphinx-doc.org

Contributing new architectures
------------------------------
If you want to contribute a new model pleas read the pages on
:ref:`architecture-life-cycle` and :ref:`adding-new-architecture`.

How to Perform a Release
-------------------------

1. **Prepare a Release Pull Request**

   - Based on the main branch create branch ``release-2025.3`` and a PR.
   - Ensure that all `CI tests <https://github.com/lab-cosmo/torch-pme/actions>`_ pass.
   - Optionally, run the tests locally to double-check.

2. **Update the Changelog**

   - Edit the changelog located in ``docs/src/dev-docs/changelog.rst``:
      - Add a new section for the new version, summarizing the changes based on the
        PRs merged since the last release.
      - Leave a placeholder section titled *Unreleased* for future updates.

3. **Merge the PR and Create a Tag**

   - Merge the release PR.
   - Update the ``main`` branch and check that the latest commit is the release PR with
     ``git log``
   - Create a tag on directly the ``main`` branch.
   - Push the tag to GitHub. For example for a release of version ``2025.3``:

     .. code-block:: bash

        git checkout main
        git pull
        git tag -a v2025.3 -m "Release v2025.3"
        git push --tags

4. **Finalize the GitHub Release**

   - Once the PR is merged, the CI will automatically:
      - Publish the package to PyPI.
      - Create a draft release on GitHub.
   - Update the GitHub release notes by pasting the changelog for the version.

5. **Merge Conda Recipe Changes**

   - May resolve and then merge an automatically created PR on the `conda recipe
     <https://github.com/conda-forge/metatrain-feedstock>`_.
   - Once thus PR is merged and the new version will be published automatically on the
     `conda-forge <https://anaconda.org/conda-forge/metatrain>`_ channel.
