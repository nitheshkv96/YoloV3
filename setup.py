from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e."

def get_requirements(requirement_txt_path:str)->List[str]:
    requirements=[]
    with open(requirement_txt_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="YoloV3",
    author = "Nithesh",
    author_email="nitheshv@umich.edu",
    version="1.0.0",
    packages = find_packages(),
    install_requires  = get_requirements('requirements.txt')
)
