#!/usr/bin/env python

"""
A set of simple proto-function to make postage stamps using fitsio
F. Menanteau, NCSA July 2015
"""
import os
import sys
import time
from collections import OrderedDict
import copy
import logging
import subprocess
import warnings

import numpy
import fitsio
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning

from des_cutter import astrometry
from des_cutter import wcsutil

SOUT = sys.stdout

# Naming template
FITS_OUTNAME = "{outdir}/{prefix}J{ra}{dec}_{filter}.{ext}"
TIFF_OUTNAME = "{outdir}/{prefix}J{ra}{dec}.{ext}"
LOG_OUTNAME = "{outdir}/{prefix}J{ra}{dec}.{ext}"
BASE_OUTNAME = "{prefix}J{ra}{dec}"
STIFF_EXE = 'stiff'

# Definitions for the color filter sets we'd like to use, by priority
# depending on what BANDS will be combined
_CSET1 = ['i', 'r', 'g']
_CSET2 = ['z', 'r', 'g']
_CSET3 = ['z', 'i', 'g']
_CSET4 = ['z', 'i', 'r']
_CSETS = (_CSET1, _CSET2, _CSET3, _CSET4)

# To avoid header warning from astropy
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

# Logger
LOGGER = logging.getLogger(__name__)


def elapsed_time(t1, verb=False):
    """
    Returns the time between t1 and the current time now
    I can can also print the formatted elapsed time.
    ----------
    t1: float
        The initial time (in seconds)
    verb: bool, optional
        Optionally print the formatted elapsed time
    returns
    -------
    stime: float
        The elapsed time in seconds since t1
    """
    t2 = time.time()
    stime = f" {int((t2 - t1) / 60)}m {(t2 - t1) - 60 * int((t2 - t1) / 60):.2f}s"
    if verb:
        print(f"Elapsed time:{stime}")
    return stime


def check_inputs(ra, dec, xsize, ysize):

    """ Check and fix inputs for cutout"""
    # Make sure that RA,DEC are the same type
    if type(ra) is not type(dec):
        raise TypeError('RA and DEC need to be the same type()')
    # Make sure that XSIZE, YSIZE are same type
    if type(xsize) is not type(ysize):
        raise TypeError('XSIZE and YSIZE need to be the same type()')
    # Make them iterable and proper length
    if hasattr(ra, '__iter__') is False and hasattr(dec, '__iter__') is False:
        ra = [ra]
        dec = [dec]
    if hasattr(xsize, '__iter__') is False and hasattr(ysize, ' __iter__') is False:
        xsize = [xsize]*len(ra)
        ysize = [ysize]*len(ra)
    # Make sure they are all of the same length
    if len(ra) != len(dec):
        raise TypeError('RA and DEC need to be the same length')
    if len(xsize) != len(ysize):
        raise TypeError('XSIZE and YSIZE need to be the same length')
    if (len(ra) != len(xsize)) or (len(ra) != len(ysize)):
        raise TypeError('RA, DEC and XSIZE and YSIZE need to be the same length')
    return ra, dec, xsize, ysize


def get_coadd_hdu_extensions_byfilename(filename):
    """
    Return the HDU extension for coadds (old-school) based on the extension name.
    Check if dealing with .fz or .fits files
    """
    if os.path.basename(os.path.splitext(filename)[-1]) == '.fz':
        sci_hdu = 1
        wgt_hdu = 2
    elif os.path.basename(os.path.splitext(filename)[-1]) == '.fits':
        sci_hdu = 0
        wgt_hdu = 1
    else:
        raise NameError("ERROR: No .fz or .fits files found")
    return sci_hdu, wgt_hdu


def get_headers_hdus(filename):
    header = OrderedDict()
    hdu = OrderedDict()

    # Case 1 -- for well-defined fitsfiles with EXTNAME
    with fitsio.FITS(filename) as fits:
        for k in range(len(fits)):
            h = fits[k].read_header()

            # Make sure that we can get the EXTNAME
            if not h.get('EXTNAME'):
                continue
            extname = h['EXTNAME'].strip()
            if extname == 'COMPRESSED_IMAGE':
                continue
            header[extname] = h
            hdu[extname] = k

    # Case 2 -- older DESDM files without EXTNAME
    if len(header) < 1:
        (sci_hdu, wgt_hdu) = get_coadd_hdu_extensions_byfilename(filename)
        fits = fitsio.FITS(filename)
        header['SCI'] = fits[sci_hdu].read_header()
        header['WGT'] = fits[wgt_hdu].read_header()
        hdu['SCI'] = sci_hdu
        hdu['WGT'] = wgt_hdu

    return header, hdu


def get_thumbFitsName(ra, dec, filter, prefix='DES', ext='fits', outdir=os.getcwd()):
    """ Common function to set the Fits thumbnail name """
    ra = astrometry.dec2deg(ra/15., sep="", plussign=False)
    dec = astrometry.dec2deg(dec, sep="", plussign=True)
    kw = locals()
    outname = FITS_OUTNAME.format(**kw)
    return outname


def get_thumbColorName(ra, dec, prefix='DES', ext='tif', outdir=os.getcwd()):
    """ Common function to set the Fits thumbnail name """
    ra = astrometry.dec2deg(ra/15., sep="", plussign=False)
    dec = astrometry.dec2deg(dec, sep="", plussign=True)
    kw = locals()
    outname = TIFF_OUTNAME.format(**kw)
    return outname


def get_thumbLogName(ra, dec, prefix='DES', ext='log', outdir=os.getcwd()):
    """ Common function to set the Fits thumbnail name """
    ra = astrometry.dec2deg(ra/15., sep="", plussign=False)
    dec = astrometry.dec2deg(dec, sep="", plussign=True)
    kw = locals()
    outname = LOG_OUTNAME.format(**kw)
    return outname


def get_thumbBaseName(ra, dec, prefix='DES'):
    """ Common function to set the Fits thumbnail name """
    ra = astrometry.dec2deg(ra/15., sep="", plussign=False)
    dec = astrometry.dec2deg(dec, sep="", plussign=True)
    kw = locals()
    outname = BASE_OUTNAME.format(**kw)
    return outname


def get_base_names(tilenames, ra, dec, prefix='DES'):
    names = []
    for k in range(len(ra)):
        if tilenames[k]:
            name = get_thumbBaseName(ra[k], dec[k], prefix=prefix)
        else:
            name = False
        names.append(name)
    return names


def update_wcs_matrix(header, x0, y0, proj='TAN'):
    """
    Update the wcs header object with the right CRPIX[1, 2] CRVAL[1, 2] for a
    given subsection

    Parameters:
    header: fits style header
        The header to work with
    x0, y0: float
        The new center of the image
    naxis1, naxis2: int
        The number of pixels on each axis.

    Returns:
        fits style header with the new center.
    """
    # We need to make a deep copy/otherwise if fails
    h = copy.deepcopy(header)
    # Get the astropy.wcs object
    wcs = WCS(h)

    if proj == 'TAN':
        # Recompute CRVAL1/2 on the new center x0,y0
        CRVAL1, CRVAL2 = wcs.wcs_pix2world(x0, y0, 1)
        # Recast numpy objects as floats
        CRVAL1 = float(CRVAL1)
        CRVAL2 = float(CRVAL2)
        # Asign CRPIX1/2 on the new image
        CRPIX1 = 1
        CRPIX2 = 1
        # Update the values
        h['CRVAL1'] = CRVAL1
        h['CRVAL2'] = CRVAL2
        h['CRPIX1'] = CRPIX1
        h['CRPIX2'] = CRPIX2
        h['CTYPE1'] = 'RA---TAN'
        h['CTYPE2'] = 'DEC--TAN'
        # Delete some key that are not needed
        dkeys = ['PROJ', 'LONPOLE', 'LATPOLE', 'POLAR', 'ALPHA0', 'DELTA0', 'X0', 'Y0']
        for k in dkeys:
            h.delete(k)

    elif proj == 'ZEA':
        CRPIX1 = float(h['CRPIX1']) - x0
        CRPIX2 = float(h['CRPIX2']) - y0
        # Delete some key that are not needed
        dkeys = ['PROJ', 'POLAR', 'ALPHA0', 'DELTA0', 'X0', 'Y0']
        for k in dkeys:
            h.delete(k)
        h['CRPIX1'] = CRPIX1
        h['CRPIX2'] = CRPIX2
        LOGGER.debug("Update to CRPIX1:%s, CRPIX2:%s", CRPIX1, CRPIX2)

    else:
        raise NameError(f"Projection: {proj} not implemented")
    return h


def fitscutter(filename, ra, dec, xsize=1.0, ysize=1.0, units='arcmin',
               prefix='DES', outdir=os.getcwd(), tilename=None, verb=False):

    """
    Makes cutouts around ra, dec for a give xsize and ysize
    ra,dec can be scalars or lists/arrays
    """
    # Check and fix inputs
    ra, dec, xsize, ysize = check_inputs(ra, dec, xsize, ysize)

    # Check for the units
    if units == 'arcsec':
        scale = 1
    elif units == 'arcmin':
        scale = 60
    elif units == 'degree':
        scale = 3600
    else:
        sys.exit("ERROR: must define units as arcses/arcmin/degree only")

    # Get header/extensions/hdu
    header, hdunum = get_headers_hdus(filename)
    extnames = header.keys()

    # Now we add the tilename to the headers -- if not already present
    if tilename and 'TILENAME' not in header['SCI']:
        if verb:
            SOUT.write(f"Will add TILENAME keyword to header for file: {filename}\n")
        tile_rec = {'name': 'TILENAME', 'value': tilename, 'comment': 'Name of DES parent TILENAME'}
        for EXTNAME in extnames:
            header[EXTNAME].add_record(tile_rec)

    # Get the pixel-scale of the input image
    pixelscale = astrometry.get_pixelscale(header['SCI'], units='arcsec')

    # Read in the WCS with wcsutil
    wcs = wcsutil.WCS(header['SCI'])

    # Extract the band/filter from the header
    if 'BAND' in header['SCI']:
        band = header['SCI']['BAND'].strip()
    elif 'FILTER' in header['SCI']:
        band = header['SCI']['FILTER'].strip()
    else:
        raise Exception("ERROR: Cannot provide suitable BAND/FILTER from SCI header")

    # Intitialize the FITS object
    ifits = fitsio.FITS(filename, 'r')

    ######################################
    # Loop over ra/dec and xsize,ysize
    for k in range(len(ra)):

        # Define the geometry of the thumbnail
        x0, y0 = wcs.sky2image(ra[k], dec[k])
        yL = 10000
        xL = 10000
        x0 = numpy.round(x0)
        y0 = numpy.round(y0)
        dx = int(0.5*xsize[k]*scale/pixelscale)
        dy = int(0.5*ysize[k]*scale/pixelscale)
        naxis1 = 2*dx
        naxis2 = 2*dy
        y1 = y0-dy
        y2 = y0+dy
        x1 = x0-dx
        x2 = x0+dx

        if y1 < 0:
            y1 = 0
            dy = y0
            y2 = y0 + dy
        if y2 > yL:
            y2 = yL
            dy = yL - y0
            y1 = y0-dy

        if x1 < 0:
            x1 = 0
            dx = x0
            x2 = x0 + dx
        if x2 > xL:
            x2 = xL
            dx = xL - x0
            x1 = x0 - dx
        im_section = OrderedDict()
        h_section = OrderedDict()
        for EXTNAME in extnames:
            # The hdunum for that extname
            HDUNUM = hdunum[EXTNAME]
            # Create a canvas
            im_section[EXTNAME] = numpy.zeros((naxis1, naxis2))
            # Read in the image section we want for SCI/WGT
            im_section[EXTNAME] = ifits[HDUNUM][int(y1):int(y2), int(x1):int(x2)]
            # Correct NAXIS1 and NAXIS2
            naxis1 = numpy.shape(im_section[EXTNAME])[1]
            naxis2 = numpy.shape(im_section[EXTNAME])[0]
            # Update the WCS in the headers and make a copy
            # h_section[EXTNAME] = update_wcs_matrix(header[EXTNAME],x0,y0,naxis1,naxis2,ra[k],dec[k])
            h_section[EXTNAME] = update_wcs_matrix(header[EXTNAME], x0, y0)
        # Construct the name of the Thumbmail using BAND/FILTER/prefix/etc
        outname = get_thumbFitsName(ra[k], dec[k], band, prefix=prefix, outdir=outdir)

        # Write out the file
        ofits = fitsio.FITS(outname, 'rw', clobber=True)
        for EXTNAME in extnames:
            ofits.write(im_section[EXTNAME], extname=EXTNAME, header=h_section[EXTNAME])

        ofits.close()
        if verb:
            SOUT.write("# Wrote: %s\n" % outname)

    return


def get_stiff_parameter_set(tiffname, **kwargs):
    """
    Set the Stiff default options and have the options to
    overwrite them with kwargs to this function.
    """
    stiff_parameters = {
        "OUTFILE_NAME": tiffname,
        "COMPRESSION_TYPE": "JPEG",
    }
    stiff_parameters.update(kwargs)
    return stiff_parameters


def make_stiff_call(fitsfiles, tiffname, stiff_parameters={}, list=False):

    """ Make the stiff call for a set of input FITS filenames"""

    pars = get_stiff_parameter_set(tiffname, **stiff_parameters)
    stiff_conf = os.path.join(os.environ['DES_CUTTER_DIR'], 'etc', 'default.stiff')

    cmd_list = []
    cmd_list.append("%s" % STIFF_EXE)
    for fname in fitsfiles:
        cmd_list.append("%s" % fname)

    cmd_list.append("-c %s" % stiff_conf)
    for param, value in pars.items():
        cmd_list.append("-%s %s" % (param, value))

    if list:
        cmd = cmd_list
    else:
        cmd = ' '.join(cmd_list)
    return cmd


def get_colorset(avail_bands, color_set=None):
    """
    Get the optimal color set for DES Survey for a set of available bands
    """
    # 1. Check if desired color_set matches the available bands """ if color_set:
    inset = list(set(color_set) & set(avail_bands))
    if len(inset) == 3:
        return color_set

    # 2. Otherwise find the optimal one
    CSET = False
    for color_set in _CSETS:
        if CSET:
            break
        inset = list(set(color_set) & set(avail_bands))
        if len(inset) == 3:
            CSET = color_set
    # 3. If no match return False
    if not CSET:
        CSET = False
    return CSET


def color_radec(ra, dec, avail_bands,
                prefix='DES', colorset=['i', 'r', 'g'],
                stiff_parameters={},
                outdir=os.getcwd(),
                verb=False):

    t0 = time.time()

    # Get colorset or match with available bands
    CSET = get_colorset(avail_bands, colorset)

    if CSET is False:
        SOUT.write("# WARNING: Could not find a suitable filter set for color image for ra,dec: %s,%s\n" % (ra, dec))
        return

    # ----------------------------------- ##
    # HERE WE COULD LOOP OVER RA,DEC if they are lists!

    # Set the output tiff name
    tiffname = get_thumbColorName(ra, dec, prefix=prefix, ext='tif', outdir=outdir)

    # Set the names of the input files
    fitsfiles = []
    for BAND in CSET:
        fitsthumb = get_thumbFitsName(ra, dec, BAND, prefix='DES', ext='fits', outdir=outdir)
        fitsfiles.append("%s" % fitsthumb)

    # Build the cmd to call
    logfile = get_thumbLogName(ra, dec, prefix=prefix, ext='stifflog', outdir=outdir)
    log = open(logfile, "w")
    cmd = make_stiff_call(fitsfiles, tiffname, stiff_parameters=stiff_parameters, list=False)
    print("RUNNING STIFF CMD:\n\t", cmd)
    status = subprocess.call(cmd, shell=True, stdout=log, stderr=log)
    if status > 0:
        SOUT.write("***\nERROR while running Stiff***\n")
    else:
        if verb:
            SOUT.write(f"# Total stiff time: {elapsed_time(t0)}\n")

    # ----------------------------------- ##

    return


if __name__ == "__main__":

    # images taken from:
    # /archive_data/Archive/OPS/coadd/20141006000032_DES0002+0001/coadd/

    # Example of inputs:
    # ra,dec can be list or scalars
    filename = 'DES0002+0001_g.fits.fz'
    ra = [0.71925223, 0.61667249, 0.615752, 0.31218133]
    dec = [0.0081421517, 0.13929069, 0.070078051, 0.08508208]
    tilename = 'DES0002+0001'

    filename = 'DES0203-0707_r2577p01_g.fits.fz'
    ra = [30.928739, 30.818148, 30.830120, 30.982164, 31.086377]
    dec = [-7.286070, -7.285457, -7.285527, -7.285317, -7.284755]

    xsize = [3]*len(ra)
    ysize = [5]*len(ra)

    t0 = time.time()
    fitscutter(filename, ra, dec, xsize=xsize, ysize=ysize, units='arcmin', prefix='DES', tilename=tilename, verb=True)
    SOUT.write(f"Done: {elapsed_time(t0)}\n")
