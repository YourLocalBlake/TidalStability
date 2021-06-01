import setuptools

version = None
with open('src/tidal_stability/__version__.py', 'r') as f:
    exec(f.read())

setuptools.setup(
    name = "tidal-stability",
    version = version,
    packages = setuptools.find_packages('src'),
    package_dir = {'': 'src'},
    install_requires = [
        "arrow>=0.17.0",
        "attrs>=20.3.0",
        "h5py>=2.10.0",
        "matplotlib>=3.3.2",
        "mpmath>=1.1.0",
        "numpy>=1.19.2",
        "scikits.odes==2.6.1",
        "scipy>=1.5.2",
        "sundials==5.4.0",
    ],
    #python_requires = '>=3.5',
    author = u"Blake Staples",
    author_email = "yourlocalblake@gmail.com",
    description = "Masters thesis code for generating solutions",
    license = "GNU version 3",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.8',
    ],
    include_package_data=True,
)