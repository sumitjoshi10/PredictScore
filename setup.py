from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    
    requirements =[]
    with open(file_path) as file:
        libraries = file.readlines()
        requirements = [name.replace("\n","") for name in libraries]
        
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
        
    return requirements

setup(
    name="Predict Score",
    version="0.0.1",
    author="Sumit Joshi",
    author_email="sumit.joshi9818@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
    )