from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setup(
    name='fasterrcnn_pytorch_training_pipeline',
    version='0.1.0',
    author='Sovit Ranjan Rath',
    author_email='sovitrath5@gmail.com',
    description='A Simple Pipeline to Train PyTorch FasterRCNN Model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/falibabaei/fasterrcnn_pytorch_training_pipeline.git',
    license='MIT',
    classifiers=[
        'Intended Audience :: Information Technology',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=requirements
)
