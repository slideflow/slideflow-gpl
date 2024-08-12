import versioneer
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slideflow_extras",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="James Dolezal",
    author_email="jamesmdolezal@gmail.com",
    description="Deep learning tools for digital histology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slideflow/slideflow_extras",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'slideflow.plugins': [
            'extras = slideflow_extras:register_extras',
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'pillow>=6.0.0',
        'gdown',
        'torch',
        'torchvision',
        'timm',
        'huggingface_hub',
        'transformers',
        'fastai',
        'scikit-misc'
    ]
)
