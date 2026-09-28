Installation Guide
==================

Basic installation
------------------

This documentation describes PADO 1.1.0. Use Python 3.10 or later
and install a PyTorch 2.10 or later build matching the intended CPU/CUDA system.
Install the 1.1.0 wheel from the GitHub release:

.. code-block:: bash

   python -m pip install "https://github.com/shwbaek/pado/releases/download/1.1.0/pado_optics-1.1.0-py3-none-any.whl"

The distribution name is ``pado-optics`` and the Python import is ``pado``.
PyPI publication is separate; an unpinned PyPI install may still select an
earlier release. See :doc:`migration` before comparing numerical results with PADO 1.0.1.

Optional dependencies
---------------------

PyTorch is the only core runtime dependency. Install extras when using their
corresponding APIs:

.. list-table::
   :header-rows: 1
   :widths: 15 45 25

   * - Extra
     - Purpose
     - Minimum
   * - ``array``
     - NumPy arrays and NPY files
     - NumPy 1.24
   * - ``viz``
     - Plotting
     - Matplotlib 3.7
   * - ``mat``
     - MAT files
     - SciPy 1.10

Install all optional APIs together:

.. code-block:: bash

   python -m pip install "pado-optics[array,viz,mat] @ https://github.com/shwbaek/pado/releases/download/1.1.0/pado_optics-1.1.0-py3-none-any.whl"

Source installation
-------------------

For development against the public repository:

.. code-block:: bash

   git clone https://github.com/shwbaek/pado.git
   cd pado
   python -m pip install -e ".[array,viz,mat]"

The installed source version follows the selected public commit; it is not
necessarily the latest published package. No Conda package availability is
claimed here.

Next steps
----------

* :doc:`migration` explains changed numerical conventions, corrections,
  gradients, precision costs and sampling limits.
* :doc:`api/index` documents the Python API.
* :doc:`examples/index` links existing examples.
* :doc:`license` describes the license.
