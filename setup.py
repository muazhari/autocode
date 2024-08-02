from setuptools import setup

setup(
    name='autocode',
    version='0.1.0',
    author='muazhari',
    url='https://github.com/muazhari',
    description='AutoCode: Automated Code Improvement by Metrics Optimization',
    packages=['autocode'],
    license='MIT',
    install_requires=[
        'pymoo',
        'pydantic_settings',
        'fastapi',
        'dependency-injector',
        'ray',
        'fastapi',
        'matplotlib<3.9.0',
        'sqlmodel',
        'dill',
        'streamlit',
        'numpy<2',
        'python-on-whales',
    ],
)
