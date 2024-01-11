import setuptools

# Read the contents of the README.md file for the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Package version
__version__ = "0.0.0"

# Repository and author information
REPO_NAME = "End-to-End-butterfly-image-classification-with-MLops"
AUTHOR_USER_NAME = "DarshanDinni"
SRC_REPO = "butterflyClassifier"
AUTHOR_EMAIL = "darshandinniul@gmail.com"

# Set up the package using setuptools
setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    # Project URLs
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
