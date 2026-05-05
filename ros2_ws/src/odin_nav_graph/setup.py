from glob import glob
import os

from setuptools import find_packages, setup
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop

package_name = 'odin_nav_graph'


# Modern setuptools (>=80) ignores the ``[install] install_scripts`` directive
# in setup.cfg, so the ament_python convention of placing the entry-point
# script in ``$base/lib/<pkg>/`` (where ``ros2 launch`` expects it) breaks.
# Force the location here regardless of setuptools version.
class _InstallWithRosScripts(_install):
    def finalize_options(self):
        super().finalize_options()
        if self.install_base:
            self.install_scripts = os.path.join(
                self.install_base, 'lib', package_name
            )


class _DevelopWithRosScripts(_develop):
    def finalize_options(self):
        super().finalize_options()
        if self.install_base:
            self.script_dir = os.path.join(
                self.install_base, 'lib', package_name
            )


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    cmdclass={'install': _InstallWithRosScripts, 'develop': _DevelopWithRosScripts},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rohan Garg',
    maintainer_email='rohan@example.com',
    description='Build a navigation graph from Odin camera point clouds using nav_graph_gpu.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nav_graph_node = odin_nav_graph.nav_graph_node:main',
        ],
    },
)
