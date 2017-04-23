from distutils.core import setup

setup(
    name='GuitarEvaluator',
    version='1.0.0',
    author='J. Nistal Hurle',
    author_email='j.nistalhurle@gmail.com',
    packages=['guitareval', 'tests'],
    scripts=['guitareval/guitar_evaluator.py'],
    url='http://pypi.python.org/pypi/GuitarEval/',
    license='LICENSE.txt',
    description='Guitar sound evaluation.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy == 1.11.0",
        "pandas == 0.19.2",
    ],
)
