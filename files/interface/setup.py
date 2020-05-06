import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lungmask",
    version="0.2.1",
    author="Johannes Hofmanninger, extended by Bernhard Kainz and Andrei Serbanescu",
    author_email="johannes.hofmanninger@meduniwien.ac.at",
    description="Package for automated lung segmentation in CT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bkainz/lungmask",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'lungmask = lungmask.__main__:main'
        ]
    },
    install_requires=[
        'pydicom',
        'numpy',
        'torch',
        'scipy',
        'SimpleITK',
        'tqdm',
        'scikit-image',
        'fill_voids'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
)