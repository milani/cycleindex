import setuptools

setuptools.setup(
    name='cycleindex',
    version='1.0',
    url='https://github.com/milani/cycleindex',
    description='To measure how unbalanced a network is, based on simple cycle counts',
    license='BSD 3-Clause',
    author='Morteza Milani',
    author_email='mrtz.milani@gmail.com',
    keywords='complex network graph heider balance index',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7'
    ],
    packages=['cycleindex'],
    python_requires='>=2.7',
    install_requires=['numpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)

