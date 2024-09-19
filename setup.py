from setuptools import find_packages, setup

#####################################
NAME = "metamol"
VERSION = "2.0.0"
ISRELEASED = False
if ISRELEASED:
    __version__ = VERSION
else:
    __version__ = VERSION + ".dev1"
#####################################

if __name__ == "__main__":

    setup(
        name=NAME,
        version=__version__,
        author="Yiqi Chen",
        author_email="yiqi.chen@metax-tech.com",
        packages=find_packages(),
        package_dir={"metamol": "metamol"},
        keywords="metamol",
        package_data={'': ['ff_files/*.xml']},
        include_package_data=True,
    )
