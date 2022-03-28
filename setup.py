import setuptools
setuptools.setup(
    name="autoseg",
    version="0.0.0",
    author="Laurel Kinman",
    author_email="lkinman@mit.edu",
    description="Statistical ",
    url="https://github.com/lkinman/autoseg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'autoseg = autoseg.__main__:main',
        ],
    },
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'pandas>=1.1.3',
        'numpy>=1.19.2',
        'networkx>=2.6.3',
        'matplotlib>=3.0.0'
        'cryodrgn>=0.2.0']

)
