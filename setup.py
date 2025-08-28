from distutils.core import setup
import glob

# The main call
setup(name='des_cutter',
      version='1.0.2',
      license="GPL",
      description="A python module to make FITS files cutouts/thumbnails for DES/DECam",
      author="Felipe Menanteau",
      author_email="felipe@illinois.edu",
      packages=['des_cutter'],
      package_dir={'': 'python'},
      scripts=['bin/des_cutter'],
      data_files=[("", ["setpath.sh"]),
                  ('etc', glob.glob("etc/*.*"))]
      )
