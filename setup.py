from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("scramble/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version


__version__ = getVersionNumber()

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/rs-station/scramble/issues",
    "Source Code": "https://github.com/rs-station/scramble",
}


LONG_DESCRIPTION = """
A simple, maximum-likelihood model for solving
the indexing ambiguity problem for laue data.
"""

setup(
    name="scramble",
    version=__version__,
    author="Kevin M. Dalton",
    author_email="kmdalton@fas.harvard.edu",
    license="MIT",
    include_package_data=True,
    packages=find_packages(),
    long_description=LONG_DESCRIPTION,
    description="It probably just randomizes your data.",
    project_urls=PROJECT_URLS,
    python_requires=">=3.8,<3.12",
    url="https://github.com/rs-station/scramble",
    install_requires=[
        "reciprocalspaceship>=0.9.16",
        "tqdm",
        "torch",
        "matplotlib",
        "seaborn",
    ],
    scripts=[
    ],
    entry_points={
        "console_scripts": [
            "scramble=scramble.scramble:main",
        ]
    },
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
)
