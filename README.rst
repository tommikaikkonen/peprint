# peprint

Drop in replacement for pprint with prettier, opinionated (but PEP8 compliant) output.

Instead of::

    from pprint import pprint

do::

    from peprint import pprint

Or to use with IPython, add these two lines to your startup file::

    from peprint import install_to_ipython
    install_to_ipython()

Packages ``colorful`` and ``pygments`` need to be installed to use ``peprint`` with ``IPython``.