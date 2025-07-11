from distutils.core import setup
import glob

# The main call
setup(name='des_cutter',
      version='0.1.0',
      license="GPL",
      description="A python module to make FITS files cutouts/thumbnails for DES",
      author="Felipe Menanteau",
      author_email="felipe@illinois.edu",
      packages=['des_cutter'],
      package_dir={'': 'python'},
      scripts=['bin/des_cutter'],
      data_files=[('etc', glob.glob("etc/*.*")),
                  ],
      )
