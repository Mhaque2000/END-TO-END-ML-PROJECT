from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(path:str) -> List[str]:
    requirements = []
    with open(path) as file:
        requirements = file.readlines()
        requirements = [r.replace('\n','') for r in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    return requirements



setup(
    name='end-to-end-ml-project',
    version='0.0.1',
    author='Mahamood',
    author_email='mahamoodulhaque@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)