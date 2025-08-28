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
import subprocess
import multiprocessing
import numpy
import fitsio
import logging
from logging.handlers import RotatingFileHandler
from astropy.utils.exceptions import AstropyWarning
from des_cutter import astrometry
from des_cutter import wcsutil
# To avoid header warning from astropy
import warnings
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

# Logger
LOGGER = logging.getLogger(__name__)

# Naming template
PREFIX = 'DECam'
OBJ_ID = "{prefix}J{ra}{dec}"
FITS_OUTNAME = "{outdir}/{objID}_{expnum}_{filter}.{ext}"
TIFF_OUTNAME = "{outdir}/{objID}_{expnum}.{ext}"
LOG_OUTNAME = "{outdir}/{objID}.{ext}"
BASE_OUTNAME = "{objID}"
BASEDIR_OUTNAME = "{outdir}/{objID}"
STIFF_EXE = 'stiff'

# Definitions for the color filter sets we'd like to use, by priority
# depending on what BANDS will be combined
_CSET1 = ['i', 'r', 'g']
_CSET2 = ['z', 'r', 'g']
_CSET3 = ['z', 'i', 'g']
_CSET4 = ['z', 'i', 'r']
_CSETS = (_CSET1, _CSET2, _CSET3, _CSET4)

# Logger
LOGGER = logging.getLogger(__name__)

try:
    DES_CUTTER_DIR = os.environ['DES_CUTTER_DIR']
except KeyError:
    DES_CUTTER_DIR = __file__.split('python')[0]


def configure_logger(logger, MP=False, logfile=None, level=logging.NOTSET,
                     log_format=None, log_format_date=None):
    """
    Configure an existing logger with specified settings. Sets the format,
    logging level, and handlers for the given logger. If a logfile is provided,
    logs are written to both the console and the file with rotation. If no log
    format or date format is provided, default values are used.

    Parameters:
    - logger (logging.Logger): The logger to configure.
    - logfile (str, optional): Path to the log file. If `None`, logs to the console.
    - level (int): Logging level (e.g., `logging.INFO`, `logging.DEBUG`).
    - log_format (str, optional): Log message format (default is detailed format with function name).
    - log_format_date (str, optional): Date format for logs (default is `'%Y-%m-%d %H:%M:%S'`).
    """
    # Define formats
    if log_format:
        FORMAT = log_format
    else:
        FORMAT = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(funcName)s] %(message)s'
    if log_format_date:
        FORMAT_DATE = log_format_date
    else:
        FORMAT_DATE = '%Y-%m-%d %H:%M:%S'

    # Update logger to see process inforation
    if MP is True:
        FORMAT = FORMAT.replace("[%(levelname)s]", "[%(levelname)s-%(processName)s]")

    formatter = logging.Formatter(FORMAT, FORMAT_DATE)

    # Need to set the root logging level as setting the level for each of the
    # handlers won't be recognized unless the root level is set at the desired
    # appropriate logging level. For example, if we set the root logger to
    # INFO, and all handlers to DEBUG, we won't receive DEBUG messages on
    # handlers.
    logger.setLevel(level)

    handlers = []
    # Set the logfile handle if required
    if logfile:
        fh = RotatingFileHandler(logfile, maxBytes=2000000, backupCount=10)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        handlers.append(fh)
        logger.addHandler(fh)

    # Set the screen handle
    if logger_has_stdout_handler(logger):
        print("Skipping adding stdout handler")
        return
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(level)
    handlers.append(sh)
    logger.addHandler(sh)
    return


def create_logger(logger=None, MP=False, logfile=None,
                  level=logging.NOTSET, log_format=None, log_format_date=None):
    """
    Configures and returns a logger with specified settings.
    Sets up logging based on provided level, format, and output file. Can be
    used for both `setup_logging` and other components.

    Parameters:
    - logger (logging.Logger, optional): The logger to configure. If `None`, a new logger
      is created.
    - logfile (str, optional): Path to the log file. If `None`, logs to the console.
    - level (int): Logging level (e.g., `logging.INFO`, `logging.DEBUG`).
    - log_format (str, optional): Format for log messages (e.g., `'%(asctime)s - %(message)s'`).
    - log_format_date (str, optional): Date format for logs (e.g., `'%Y-%m-%d %H:%M:%S'`).

    Returns:
    logging.Logger: The configured logger instance.

    Raises:
    - ValueError: If the log level or format is invalid.
    """

    if logger is None:
        logger = logging.getLogger(__name__)
    configure_logger(logger, MP=MP, logfile=logfile, level=level,
                     log_format=log_format, log_format_date=log_format_date)
    logging.basicConfig(handlers=logger.handlers, level=level)
    logger.propagate = False
    logger.info(f"Logging Created at level:{level}")
    if logfile:
        logger.info(f"Will write log to file: {logfile}")
    return logger


def logger_has_stdout_handler(logger):
    """
    Check whether the given logger has a StreamHandler that
    writes to sys.stdout.
    Parameters:
     logger (logging.Logger): The logger instance to inspect.
    Returns:
     bool: True if a StreamHandler writing to sys.stdout is found,
           False otherwise.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            return True
    return False


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


def check_inputs(ra, dec, xsize, ysize, objID=None):

    """ Check and fix inputs for cutout"""
    # Make sure that RA,DEC are the same type
    if type(ra) is not type(dec):
        raise TypeError('RA and DEC need to be the same type()')

    # Same for objID
    if objID is not None and type(objID) is not type(ra):
        raise TypeError('objID needs to be the same type() as RA,DEC')

    # Make sure that XSIZE, YSIZE are same type
    if type(xsize) is not type(ysize):
        raise TypeError('XSIZE and YSIZE need to be the same type()')
    # Make them iterable and proper length
    if hasattr(ra, '__iter__') is False and hasattr(dec, '__iter__') is False:
        ra = numpy.asarray([ra])
        dec = numpy.asarray([dec])
    if objID is not None and hasattr(objID, '__iter__') is False:
        objID = [objID]
    if hasattr(xsize, '__iter__') is False and hasattr(ysize, ' __iter__') is False:
        xsize = numpy.asarray([xsize]*len(ra))
        ysize = numpy.asarray([ysize]*len(ra))
    # Make sure they are all of the same length
    if len(ra) != len(dec):
        raise TypeError('RA and DEC need to be the same length')
    if objID is not None and len(objID) != len(ra):
        raise TypeError('objID needs to be the same length as RA, DEC')
    if len(xsize) != len(ysize):
        raise TypeError('XSIZE and YSIZE need to be the same length')
    if (len(ra) != len(xsize)) or (len(ra) != len(ysize)):
        raise TypeError('RA, DEC and XSIZE and YSIZE need to be the same length')
    # Now make sure that all objID are unique
    if objID is not None and len(set(objID)) != len(objID):
        raise TypeError('Elements in objID are not unique')
    # If objID is None, return a list of None of the same lenght as ra,dec
    if objID is None:
        objID = [objID]*len(ra)
    return ra, dec, xsize, ysize, objID


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

    # Fix Nones
    for key in header['SCI'].keys():
        if key is None:
            header['SCI'].delete(key)

    for key in header['WGT'].keys():
        if key is None:
            header['WGT'].delete(key)

    return header, hdu


def get_thumbFitsName(ra, dec, filter, expnum='coadd',
                      objID=None, prefix=PREFIX, ext='fits', outdir=os.getcwd()):
    """ Common function to set the Fits thumbnail name """
    ra = astrometry.dec2deg(ra/15., sep="", plussign=False)
    dec = astrometry.dec2deg(dec, sep="", plussign=True)
    if objID is None:
        objID = OBJ_ID.format(ra=ra, dec=dec, prefix=prefix)
    # Locals need to be captured at the end
    kw = locals()
    outname = FITS_OUTNAME.format(**kw)
    return outname


def get_thumbBaseDirName(ra, dec, objID=None, prefix=PREFIX, outdir=os.getcwd()):
    """ Common function to set the Fits thumbnail name """
    ra = astrometry.dec2deg(ra/15., sep="", plussign=False)
    dec = astrometry.dec2deg(dec, sep="", plussign=True)
    if objID is None:
        objID = OBJ_ID.format(ra=ra, dec=dec, prefix=prefix)
    # Locals need to be captured at the end
    kw = locals()
    basedir = BASEDIR_OUTNAME.format(**kw)
    return basedir


def get_thumbColorName(ra, dec, objID=None, expnum='coadd', prefix=PREFIX, ext='tif', outdir=os.getcwd()):
    """ Common function to set the Fits thumbnail name """
    ra = astrometry.dec2deg(ra/15., sep="", plussign=False)
    dec = astrometry.dec2deg(dec, sep="", plussign=True)
    if objID is None:
        objID = OBJ_ID.format(ra=ra, dec=dec, prefix=prefix)
    kw = locals()
    outname = TIFF_OUTNAME.format(**kw)
    return outname


def get_thumbLogName(ra, dec, objID=None, prefix=PREFIX, ext='log', outdir=os.getcwd()):
    """ Common function to set the Fits thumbnail name """
    ra = astrometry.dec2deg(ra/15., sep="", plussign=False)
    dec = astrometry.dec2deg(dec, sep="", plussign=True)
    if objID is None:
        objID = OBJ_ID.format(ra=ra, dec=dec, prefix=prefix)
    # Locals need to be captured at the end
    kw = locals()
    outname = LOG_OUTNAME.format(**kw)
    return outname


def get_thumbBaseName(ra, dec, objID=None, prefix=PREFIX):
    """ Common function to set the Fits thumbnail name """
    ra = astrometry.dec2deg(ra/15., sep="", plussign=False)
    dec = astrometry.dec2deg(dec, sep="", plussign=True)
    if objID is None:
        objID = OBJ_ID.format(ra=ra, dec=dec, prefix=prefix)
    # Locals need to be captured at the end
    kw = locals()
    outname = BASE_OUTNAME.format(**kw)
    return outname


def get_base_names(tilenames, ra, dec, prefix=PREFIX):
    names = []
    for k in range(len(ra)):
        if tilenames[k]:
            name = get_thumbBaseName(ra[k], dec[k], prefix=prefix)
        else:
            name = False
        names.append(name)
    return names


def update_wcs_matrix(header, x0, y0, naxis1, naxis2, proj='TAN'):
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

    proj: the desired projection (TAN/TVP/ZEA)
    Returns:
        fits style header with the new center.
    """
    # We need to make a deep copy/otherwise if fails
    h = copy.deepcopy(header)

    if proj == 'TAN':
        CRPIX1 = float(h['CRPIX1']) - x0 + naxis1/2.
        CRPIX2 = float(h['CRPIX2']) - y0 + naxis2/2.
        h['CRPIX1'] = CRPIX1
        h['CRPIX2'] = CRPIX2
        h['CTYPE1'] = 'RA---TAN'
        h['CTYPE2'] = 'DEC--TAN'
        # Delete some key that are not needed
        dkeys = ['PROJ', 'LONPOLE', 'LATPOLE', 'POLAR', 'ALPHA0', 'DELTA0', 'X0', 'Y0']
        for k in dkeys:
            h.delete(k)

    elif proj == 'TPV':
        CRPIX1 = float(h['CRPIX1']) - x0 + naxis1/2.
        CRPIX2 = float(h['CRPIX2']) - y0 + naxis2/2.
        h['CRPIX1'] = CRPIX1
        h['CRPIX2'] = CRPIX2
        h['CTYPE1'] = 'RA---TPV'
        h['CTYPE2'] = 'DEC--TPV'
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


def get_NP(MP):

    """ Get the number of processors in the machine
    if MP == 0, use all available processor
    """
    # For it to be a integer
    MP = int(MP)
    if MP == 0:
        NP = int(multiprocessing.cpu_count())
    elif isinstance(MP, int):
        NP = MP
    else:
        raise ValueError('MP is wrong type: %s, integer type' % MP)
    return NP


def fitscutter(filename, ra, dec, objID=None, xsize=1.0, ysize=1.0, units='arcmin',
               prefix=PREFIX, outdir=None, tilename=None, counter='', logger=None):

    """
    Makes cutouts around ra, dec for a give xsize and ysize
    ra,dec can be scalars or lists/arrays

    Input Parameters
    ----------------
    filename: string
        The fitsfile (fits or fits.fz) to cut from
    ra, dec: float or array of floats
        The position in decimal degrees where we can to cut

    Input/Output Parameters
    -----------------------
    These are dictionary that need to be inputs to be passed back when
    running using multiprocessing

    cutout_names: dictionary
        Dict story the names of the cutout files
    rejected_names: dictionary
        Dict of rejected ids per filename with centers outside the FITS files
        or WGT=0
    lightcurve: dictionary
        Dict with lightcurve information

    Optional inputs
    ---------------
    xsize, ysize: float or array of floats
        The x,y size of the stamps in arcmins
    objID: string or list of string
        The list of object ID for each ra, dec pair
    units: string
        The units for the xsise, ysize
    prefix: string
        The prefix to prepend to the output names (i.e.: spt3g)
    outdir: string
        The path for the output directory
    clobber: Bool
        Overwrite file if they exist
    logger: logging object
        Optional logging object
    counter: string
        Optional counter to pass on for tracking flow

    Returns:
        cutout_names
    """
    # global timer for function
    t1 = time.time()
    if not logger:
        logger = LOGGER
    if not outdir:
        outdir = os.getcwd()

    # Check and fix inputs
    ra, dec, xsize, ysize, objID = check_inputs(ra, dec, xsize, ysize, objID)
    logger.info(f"Will cut: {len(ra)} stamps from FITS file: {filename} -- {counter}")

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
    t0 = time.time()
    header, hdunum = get_headers_hdus(filename)
    logger.debug(f"Done Getting header, hdus: {elapsed_time(t0)}")
    extnames = header.keys()
    logger.debug(f"Found EXTNAMES:{extnames}")

    # Now we add the tilename to the headers -- if not already present
    if tilename and 'TILENAME' not in header['SCI']:
        logger.info(f"Will add TILENAME keyword to header for file: {filename}\n")
        tile_rec = {'name': 'TILENAME', 'value': tilename, 'comment': 'Name of DES parent TILENAME'}
        for EXTNAME in extnames:
            header[EXTNAME].add_record(tile_rec)

    # Get the pixel-scale of the input image
    pixelscale = astrometry.get_pixelscale(header['SCI'], units='arcsec')
    # Read in the WCS with wcsutil
    wcs = wcsutil.WCS(header['SCI'])

    # Get the dimensions of the parent image
    if 'EXTNAME' in header['SCI'] and header['SCI']['EXTNAME'].strip() == 'COMPRESSED_IMAGE':
        NAXIS1 = header['SCI']['ZNAXIS1']
        NAXIS2 = header['SCI']['ZNAXIS2']
    elif 'ZIMAGE' in header['SCI'] and header['SCI']['ZIMAGE'] is True:
        NAXIS1 = header['SCI']['ZNAXIS1']
        NAXIS2 = header['SCI']['ZNAXIS2']
    else:
        NAXIS1 = header['SCI']['NAXIS1']
        NAXIS2 = header['SCI']['NAXIS2']

    # Extract the band/filter from the header
    if 'BAND' in header['SCI']:
        band = header['SCI']['BAND'].strip()
    elif 'FILTER' in header['SCI']:
        band = header['SCI']['FILTER'].strip()
    else:
        raise Exception("ERROR: Cannot provide suitable BAND/FILTER from SCI header")

    if 'EXPNUM' in header['SCI']:
        expnum = str(header['SCI']['EXPNUM']).strip()
    else:
        expnum = 'coadd'

    if 'CTYPE1' in header['SCI']:
        if header['SCI']['CTYPE1'].endswith("-TAN"):
            proj = 'TAN'
        elif header['SCI']['CTYPE1'].endswith("-TPV"):
            proj = 'TPV'
        else:
            raise NameError(f"Projection: {proj} not implemented")
        logger.debug(f"Will use prpjection: {proj} for {filename}")
    else:
        logger.debug(f"CTYPE1 not found for {filename}")

    # Intitialize the FITS object
    t0 = time.time()
    ifits = fitsio.FITS(filename, 'r')
    logger.debug(f"Done loading fitsio.FITS({filename}): {elapsed_time(t0)}")

    ######################################
    # Loop over ra/dec and xsize,ysize
    for k in range(len(ra)):

        # The basename for the (ra,dec)
        if objID[k] is None:
            objID[k] = get_thumbBaseName(ra[k], dec[k], prefix=prefix)
        logger.debug(f"Defined objID: {objID[k]}")

        # image and header sections
        im_section = OrderedDict()
        h_section = OrderedDict()

        # Define the geometry of the thumbnail
        x0, y0 = wcs.sky2image(ra[k], dec[k])
        x0 = round(float(x0))
        y0 = round(float(y0))
        dx = int(0.5*xsize[k]*scale/pixelscale)
        dy = int(0.5*ysize[k]*scale/pixelscale)
        naxis1 = 2*dx  # +1
        naxis2 = 2*dy  # +1
        y1 = y0-dy
        y2 = y0+dy
        x1 = x0-dx
        x2 = x0+dx

        # Make sure the (x0,y0) is contained within the image
        if x0 < 0 or y0 < 0 or x0 > NAXIS1 or y0 > NAXIS2:
            logger.warning(f"Rejected {objID[k]} (RA,DEC):{ra[k]},{dec[k]} outside {filename}")
            logger.warning(f"Rejected {objID[k]} (x0,y0):{x0},{y0} > {NAXIS1},{NAXIS2}")
            # rejected_ids.append(objID[k])
            continue

        # Make sure we are not going beyond the limits
        # if negative set it to zero
        if y1 < 0:
            y1 = 0
        if y2 > NAXIS2:
            y2 = NAXIS2
        if x1 < 0:
            x1 = 0
        if x2 > NAXIS1:
            x2 = NAXIS1

        logger.debug(f"Working on object:{k} -- {objID[k]}")
        logger.debug(f"Found naxis1,naxis2: {naxis1},{naxis2}")
        logger.debug(f"Found x0,x1,x2: {x0},{x1},{x2}")
        logger.debug(f"Found y0,y1,y2: {y0},{y1},{y2}")

        # Now we cut the fits stamp -- loop over HDUs/EXTNAMES
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
            h_section[EXTNAME] = update_wcs_matrix(header['SCI'], x0, y0, naxis1, naxis2, proj=proj)
            # Add the objID to the header of the thumbnail
            rec = {'name': 'OBJECT', 'value': objID[k], 'comment': 'Name of the objID'}
            h_section[EXTNAME].add_record(rec)

        # Get the basedir
        basedir = get_thumbBaseDirName(ra[k], dec[k], objID=objID[k], prefix=prefix, outdir=outdir)
        if not os.path.exists(basedir):
            os.makedirs(basedir, mode=0o755, exist_ok=True)
        # Construct the name of the Thumbmail using BAND/FILTER/prefix/etc
        outname = get_thumbFitsName(ra[k], dec[k], band, objID=objID[k],
                                    expnum=expnum, prefix=prefix, outdir=basedir)
        # Write out the file
        ofits = fitsio.FITS(outname, 'rw', clobber=True)
        for EXTNAME in extnames:
            ofits.write(im_section[EXTNAME], extname=EXTNAME, header=h_section[EXTNAME])

        ofits.close()
        logger.info(f"Done writing {outname}: {elapsed_time(t0)}")

    ifits.close()
    logger.info(f"Done filename: {filename} in {elapsed_time(t1)} -- {counter}")
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
    stiff_conf = os.path.join(DES_CUTTER_DIR, 'etc', 'default.stiff')

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
                objID=None,
                prefix=PREFIX, colorset=['i', 'r', 'g'],
                stiff_parameters={},
                outdir=os.getcwd(),
                verb=False):

    t0 = time.time()

    # Get colorset or match with available bands
    CSET = get_colorset(avail_bands, colorset)

    if CSET is False:
        LOGGER.warning(f"Could not find a suitable filter set for color image for ra,dec: {ra},{dec}")
        return

    # ----------------------------------- ##
    # HERE WE COULD LOOP OVER RA,DEC if they are lists!

    # Set the output tiff name
    basedir = get_thumbBaseDirName(ra, dec, objID=objID, prefix=prefix, outdir=outdir)
    tiffname = get_thumbColorName(ra, dec, prefix=prefix, ext='tiff', outdir=basedir)

    # Set the names of the input files
    fitsfiles = []
    for BAND in CSET:
        fitsthumb = get_thumbFitsName(ra, dec, BAND, prefix=PREFIX, ext='fits', outdir=basedir)
        fitsfiles.append("%s" % fitsthumb)

    # Build the cmd to call
    cmd = make_stiff_call(fitsfiles, tiffname, stiff_parameters=stiff_parameters, list=False)
    LOGGER.info(f"RUNNING STIFF CMD:\n{cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        LOGGER.info(result.stdout.strip())
    if result.stderr:
        LOGGER.error(result.stderr.strip())

    if result.returncode > 0:
        LOGGER.error("ERROR while running Stiff")
    else:
        LOGGER.info(f"Total stiff time: {elapsed_time(t0)}")
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
    fitscutter(filename, ra, dec, xsize=xsize, ysize=ysize, units='arcmin', prefix='DES', tilename=tilename)
    print(f"Done: {elapsed_time(t0)}")
