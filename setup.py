import setuptools
from pathlib import Path


def get_long_description() -> str:
    return Path("README.md").read_text(encoding="utf8")


with open('src/tidal_stability/__version__.py', 'r') as f:
    version = None
    exec(f.read())

setuptools.setup(
    name="tidal-stability",
    version=version,
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        "arrow>=0.17.0",
        "attrs>=20.3.0",
        "h5py>=2.10.0",
        "matplotlib>=3.3.2",
        "mpmath>=1.1.0",
        "numpy>=1.19.2",
        "scipy>=1.5.2",
        "scikits.odes",  # Shipped without odes - user to install
    ],
    python_requires='>=3.6',
    author="Blake Staples",
    author_email="yourlocalblake@gmail.com",
    description="Solver for gas clouds around Black Holes",
    long_description_content_type="text/markdown",
    long_description=get_long_description(),
    license="GPLv3",
    url="https://github.com/YourLocalBlake/TidalStability",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    include_package_data=True,
)
