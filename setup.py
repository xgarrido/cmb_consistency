from setuptools import setup

setup(
    name="consistency",
    version="0.1",
    description="Check CMB consistency across different assumptions",
    author="Xavier Garrido",
    author_email="xavier.garrido@gmail.com",
    packages=["likelihoods"],
    python_requires=">=3.5",
    include_package_data=True,
    install_requires=[
        "cobaya>=3.0.4",
        "camb",
        "spt @ git+https://github.com/xgarrido/spt_likelihoods@master#egg=spt",
    ],
)
