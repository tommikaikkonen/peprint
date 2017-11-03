=====
Usage
=====

You may use peprint as a drop-in replacement for ``pprint``. Instead of import ``pprint``, import ``peprint``::

    from peprint import pprint
    pprint({'a': 1, 'b': True})

You can use it as the output printer in the IPython shell::
    
    >>> from peprint import install_to_ipython
    >>> install_to_ipython()
    >>> {'a': 1, 'b': True}

If you want to use peprint as the default printer, you may add the first two lines to your IPython startup files. The syntax highlighting used in IPython will be used to highlight the output.