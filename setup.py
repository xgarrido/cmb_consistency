from setuptools import setup

setup(
    name="consistency",
    version="0.1",
    description="Check CMB consistency across different assumptions",
    author="Xavier Garrido",
    author_email="xavier.garrido@gmail.com",
    packages=["likelihoods"],
    python_requires=">3",
    include_package_data=True,
    install_requires=["cobaya"],
)
