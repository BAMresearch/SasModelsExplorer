from setuptools import setup, find_packages

setup(
    name='SasModelsExplorer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'PyQt5',
        'sasmodels',
        'numpy',
        'matplotlib',
        'pint',
        # other dependencies as listed in requirements.txt
    ],
    entry_points={
        'console_scripts': [
            'ModelExplorer = ModelExplorer.__main__:main',
        ],
        'gui_scripts': []  # if your app is GUI only and you want certain scripts for cross-platform GUI behavior
    },
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.rst'],  # include all text and rst files, change the pattern according to your need
    },
    description='A PyQt-based interactive application for exploring scattering models using sasmodels library.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)