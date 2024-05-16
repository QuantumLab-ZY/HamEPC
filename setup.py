'''
Descripttion: 
version: 
Author: Yang Zhong & Shixu Liu
Date: 2024-05-16 21:32:35
LastEditors: Yang Zhong
LastEditTime: 2024-05-16 23:41:55
'''


from setuptools import setup, find_packages

setup(
    name='HamEPC',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torch_geometric",
        "e3nn",
        "pymatgen",
        "tqdm",
        "natsort",
        'pyyaml',
        'mpi4py'
    ],
    entry_points={
        'console_scripts': [
            'HamEPC=HamEPC.run_EPC:main'
        ]
    },
    author='Yang Zhong & Shixu Liu',
    author_email='yzhong@fudan.edu.cn',
    description='A machine learning workflow based on HamGNN for calculating the electron-phonon coupling (EPC).',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/QuantumLab-ZY/HamEPC.git',  # Replace with the actual URL of your project
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)