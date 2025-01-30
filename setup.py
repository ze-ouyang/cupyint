from setuptools import setup, find_packages

setup(
    name="cupyint",  
    version="0.1.7", 
    author="Ze Ouyang", 
    author_email="ze_ouyang@utexas.edu",  
    description="A CuPy-based numerical integration library", 
    long_description=open("README.md").read(), 
    long_description_content_type="text/markdown",  
    url="https://github.com/ze-ouyang/cupyint",  
    packages=find_packages(),  
    install_requires=[  
        #"cupy","numpy",
    ],
    classifiers=[  
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)