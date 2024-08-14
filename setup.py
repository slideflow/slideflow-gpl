import versioneer
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slideflow-gpl",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="James Dolezal",
    author_email="james@slideflow.ai",
    description="GPL-3 extensions and tools for Slideflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slideflow/slideflow-gpl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'slideflow.plugins': [
            'extras = slideflow_gpl:register_extras',
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        'slideflow>=3.0'
        'numpy',
        'pillow>=6.0.0',
        'gdown',
        'torch',
        'torchvision',
        'timm',
        'huggingface_hub',
        'transformers',
        'fastai',
        'scikit-misc',
    ]
)
