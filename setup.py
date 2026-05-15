from setuptools import find_packages, setup

package_name = 'my_policy_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['my_policy_node/config/attempt1_config.yaml']),

    ],
    package_data={
        'my_policy_node': [
            'config/*.yaml',
            'preprocess/a_image_processing/*.pt',
        ],
        '': ['py.typed'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dimas Ridhwana S',
    maintainer_email='dimsridhwana@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
