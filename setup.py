from distutils.core import setup

setup(
    name='pandora-prescriptor',
    version='0.1.0',
    packages=['pandora', 'covid_xprize'],
    url='https://github.com/rbilleci/pandora-prescriptor',
    license='',
    author='Richard Billeci',
    author_email='rick.billeci@gmail.com',
    description='',
    package_data={
        "pandora": ["data/*.*"]  # include all files in the data directory
    },
    install_requires=[
        'pandas~=1.2.1',
        'keras~=2.4.3',
        'tensorflow~=2.4.1',
        'numpy~=1.19.5',
        'neat-python~=0.92']
)
