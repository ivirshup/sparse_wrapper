import setuptools

setuptools.setup(
    name="sparse_wrapper",
    description="Array wrapper for scipy.sparse matrices",
    url="https://github.com/ivirshup/sparse_wrapper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
