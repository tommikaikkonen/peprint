peprint
-------

Drop in replacement for pprint with prettier, opinionated (but PEP8 compliant) output.

Usage
-----

This package is not available on PyPi yet. Install directly from the repo:

.. code:: bash
    
    pip install -e git+https://github.com/tommikaikkonen/peprint.git#egg=peprint

Then, instead of

.. code:: python

    from pprint import pprint

do

.. code:: python

    from peprint import pprint


Usage with IPython
------------------

You can use peprint with IPython so that values in the REPL will be printed with ``peprint`` using syntax highlighting. You need to call ``peprint`` initialization functions at the start of an IPython session, which IPython facilitates with `profile startup files`_. To initialize peprint in your default profile, add and edit a new startup file with the following commands:

.. code:: bash
    
    touch "`ipython locate profile default`/startup/init_peprint.py"
    nano "`ipython locate profile default`/startup/init_peprint.py"


The code in this file will be run upon entering the shell. Then add one or more of these lines:

.. code:: python

    # Use peprint as the default pretty printer in IPython
    import peprint.extras.ipython

    peprint.extras.ipython.install()

    # Specify syntax higlighting theme in IPython;
    # will be picked up by peprint.
    from pygments import styles

    ipy = get_ipython()
    ipy.colors = 'linux'
    ipy.highlighting_style = styles.get_style_by_name('monokai')


    # For Django users: install prettyprinter for Django models and QuerySets.
    import peprint.extras.django
    peprint.extras.django.install()

Packages colorful_ and pygments_ need to be installed to use ``peprint`` with ``IPython``.

.. _`profile startup files`: http://ipython.readthedocs.io/en/stable/config/intro.html#profiles
.. _colorful: https://github.com/timofurrer/colorful
.. _pygments: https://pypi.python.org/pypi/Pygments