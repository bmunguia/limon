# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['pymeshb', 'pymeshb.gamma']

package_data = \
{'': ['*']}

install_requires = \
['docopt==0.6.2',
 'matplotlib==3.9.2',
 'numpy>=2.0,<3.0',
 'pybind11==2.13.6',
 'pyyaml==6.0.2',
 'vtk==9.3.1']

setup_kwargs = {
    'name': 'pymeshb',
    'version': '0.1.0',
    'description': 'Python API to handle the *.meshb file format',
    'long_description': '# pymeshb\nPython API to handle the *.meshb file format\n',
    'author': 'bmunguia',
    'author_email': 'bmunguia@stanford.edu',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://github.com/bmunguia/pymeshb.git',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '~=3.12',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
