Installation
=========================

Prerequisites
----------------------------

  * Python 3.6 to 3.12
  * numpy, scipy, matplotlib, pyqt5, h5py, configparser, schema
  * daetools

Install via pip
-----------------------------

The latest version MPET can be installed directly from Github with the following command:

``pip install git+https://github.com/TRI-AMDD/mpet.git``

Install from source
----------------------------

You can also download the source code and install the latest version

 * clone the repository : ``git clone https://github.com/TRI-AMDD/mpet.git``
 * Enter the mpet directory : ``cd mpet``
 * install MPET using pip ``pip install -e .``

MPET on Linux
-----------------------------

Running MPET on Linux requires a few additional libraries, which are dependencies for daetools:

.. code-block:: bash

  apt-get install libgl1-mesa-glx libgfortran5

Test your installation
---------------------------
 To test your installation make sure to install MPET following :

 ``pip install -e .[test]``

 
 Then you can run the tests with

 ``./bin/mpettest.py``

Common Installation Bugs
---------------------------

One of the most common bugs is with QT plugins (it is not acutally a problem with MPET, but with one of the packages that MPET uses). The bug will usually cause plots to not be able to initialize and have the following error message:


.. code-block:: RST

    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
    Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl, xcb.
    Aborted (core dumped)``

If you get this bug, first, check to see that your X11 server is in use!
If not, try turning on debugging for QT plugins with 

``export QT_DEBUG_PLUGINS=1``. 

Often, the library issue that appears is 


.. code-block:: RST

    Cannot load library /.../lib/python3.7/site-packages/PyQt5/Qt5/plugins/platforms/libqxcb.so:
    (libxcb-xinerama.so.0: cannot open shared object file: No such file or directory)``.

If this is the issue, outside of an virtual environment, install the libxcb-xinerama0 package with

``sudo apt-get install libxcb-xinerama0``.
