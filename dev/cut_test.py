#!/usr/bin/env python

from astropy.wcs import WCS
import logging
import copy
import fitsio
from astropy.nddata import Cutout2D
from astropy.io import fits
from des_cutter import thumbslib
from des_cutter import wcsutil
from des_cutter import astrometry
import astropy


# Logger
LOGGER = logging.getLogger(__name__)


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

    Returns:
        fits style header with the new center.
    """
    # We need to make a deep copy/otherwise if fails
    h = copy.deepcopy(header)
    # Get the astropy.wcs object
    # wcs = WCS(h)
    # Get the wcsutil wcs object
    wcs = wcsutil.WCS(h)

    if proj == 'TAN':
        # Recompute CRVAL1/2 on the new center x0,y0
        # CRVAL1, CRVAL2 = wcs.wcs_pix2world(x0, y0, 1)
        CRVAL1, CRVAL2 = wcs.image2sky(x0, y0)
        # Recast numpy objects as floats
        CRVAL1 = float(CRVAL1)
        CRVAL2 = float(CRVAL2)
        # Asign CRPIX1/2 on the new image
        CRPIX1 = int(naxis1/2.)
        CRPIX2 = int(naxis2/2.)
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

    elif proj == 'TPV':
        CRPIX1 = float(h['CRPIX1']) - x0 + naxis1/2.
        CRPIX2 = float(h['CRPIX2']) - y0 + naxis2/2.
        h['CRPIX1'] = CRPIX1
        h['CRPIX2'] = CRPIX2
        h['CTYPE1'] = 'RA---TPV'
        h['CTYPE2'] = 'DEC--TPV'

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


def cutter(filename, ra, dec, xsize, ysize, units='arcmin'):

    # Check for the units
    if units == 'arcsec':
        scale = 1
    elif units == 'arcmin':
        scale = 60
    elif units == 'degree':
        scale = 3600
    else:
        raise ("ERROR: must define units as arcses/arcmin/degree only")

    header, hdunum = thumbslib.get_headers_hdus(filename)

    # Get the pixel-scale of the input image
    pixelscale = astrometry.get_pixelscale(header['SCI'], units='arcsec')

    # Get the wcs object
    # wcs = WCS(header['SCI'], relax=True)
    # Read in the WCS with wcsutil
    wcs_util = wcsutil.WCS(header['SCI'])
    wcs_astropy = astropy.wcs.WCS(header['SCI'], relax=True)

    x0, y0 = wcs_util.sky2image(ra, dec)
    x0, y0 = wcs_astropy.wcs_world2pix(ra, dec, 1)
    x0 = round(float(x0))
    y0 = round(float(y0))
    dx = int(0.5*xsize*scale/pixelscale)
    dy = int(0.5*ysize*scale/pixelscale)
    naxis1 = 2*dx  # +1
    naxis2 = 2*dy  # +1
    y1 = y0 - dy
    y2 = y0 + dy
    x1 = x0 - dx
    x2 = x0 + dx

    outfile_astropy = f"stamp_astropy_{filename}"

    # Cut with astropy
    with fits.open(filename) as hdul:
        hdu = hdul['SCI']
        cut = Cutout2D(hdu.data, (x0, y0), (naxis1, naxis2), wcs=wcs_astropy, mode='strict')
        # Start from the original header
        hdr = hdu.header.copy()
        hdr.update(cut.wcs.to_header(relax=True))
        hdu_out = fits.PrimaryHDU(cut.data, header=hdr)
        hdu_out.writeto(outfile_astropy, overwrite=True)
        print(f"Wrote: {outfile_astropy}")

    # Cut with fitsio
    outfile_fitsio = f"stamp_fitsio_{filename}"
    ifits = fitsio.FITS(filename, 'r')
    im_section = ifits['SCI'][int(y1):int(y2), int(x1):int(x2)]
    hdr = update_wcs_matrix(header['SCI'], x0, y0, naxis1, naxis2, proj='TPV')
    ofits = fitsio.FITS(outfile_fitsio, 'rw', clobber=True)
    ofits.write(im_section, extname='SCI', header=hdr)
    ofits.close()
    print(f"Wrote: {outfile_fitsio}")
    return


if __name__ == "__main__":
    # Define position and size
    ra = 15.71875
    dec = -49.24944
    xsize = 1
    ysize = 1

    filename = 'DES0102-4914_r4907p01_i.fits.fz'
    cutter(filename, ra, dec, xsize, ysize)
    filename = 'D00252231_i_c18_r3651p01_immasked.fits.fz'
    cutter(filename, ra, dec, xsize, ysize)
