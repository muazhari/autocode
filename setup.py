from pathlib import Path

from setuptools import setup

this_directory: Path = Path(__file__).parent
long_description: str = (this_directory / "README.md").read_text()

setup(
    name='autocode-py',
    version='0.0.1.post9',
    author='muazhari',
    url='https://github.com/muazhari/autocode',
    description='autocode: Auto Code Improvement by Metrics Optimization.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['autocode'],
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
    ],
    install_requires=[
        'pymoo@git+https://github.com/anyoptimization/pymoo.git',
        'pydantic_settings',
        'fastapi',
        'dependency-injector>=4.42.0b1',
        'ray',
        'fastapi',
        'matplotlib<3.9.0',
        'sqlmodel',
        'dill',
        'streamlit',
        'numpy>=1.26.4,<2',
        'python-on-whales',
        'uvicorn',
        'langchain',
        'langchain-openai>=0.1.21rc1',
        'langgraph',
    ],
)
