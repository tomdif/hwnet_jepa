from setuptools import setup, find_packages

setup(
    name="hwnet_jepa",
    version="0.1.0",
    description="Hubel-Wiesel-inspired biological vision frontend with I-JEPA self-supervised pretraining",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "matplotlib>=3.6",
    ],
    python_requires=">=3.9",
)
