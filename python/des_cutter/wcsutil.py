import math
import sys
"""

 Taken from Erin Sheldon's esutil package at:
 https://github.com/esheldon/esutil/blob/master/esutil/wcsutil.py
 2020

Module:
    wcsutil

Contains the class WCS to perform world coordinate system transformations.
See documentation for the WCS class for more information.

"""

_license = """
  Copyright (C) 2010  Erin Sheldon

    This program is free software; you can redistribute it and/or modify it
    under the terms of version 2 of the GNU General Public License as
    published by the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""


try:
    import numpy
    from numpy import isscalar
    import scipy.optimize
    from scipy.optimize import leastsq
    have_numpy = True
except KeyError:
    have_numpy = False

r2d = 180.0 / math.pi
d2r = math.pi / 180.0
DEFTOL = 1e-8

# map the odd scamp naming scheme onto a matrix
# I didn't figure out the formula

_scamp_max_order = 3
_scamp_max_ncoeff = 11
_scamp_skip = [3]

_scamp_map = {}
_scamp_map['pv1_0'] = (0, 0)
_scamp_map['pv1_1'] = (1, 0)
_scamp_map['pv1_2'] = (0, 1)
_scamp_map['pv1_4'] = (2, 0)
_scamp_map['pv1_5'] = (1, 1)
_scamp_map['pv1_6'] = (0, 2)
_scamp_map['pv1_7'] = (3, 0)
_scamp_map['pv1_8'] = (2, 1)
_scamp_map['pv1_9'] = (1, 2)
_scamp_map['pv1_10'] = (0, 3)

_scamp_map['pv2_0'] = (0, 0)
_scamp_map['pv2_1'] = (0, 1)
_scamp_map['pv2_2'] = (1, 0)
_scamp_map['pv2_4'] = (0, 2)
_scamp_map['pv2_5'] = (1, 1)
_scamp_map['pv2_6'] = (2, 0)
_scamp_map['pv2_7'] = (0, 3)
_scamp_map['pv2_8'] = (1, 2)
_scamp_map['pv2_9'] = (2, 1)
_scamp_map['pv2_10'] = (3, 0)

_allowed_projections = ['-TAN', '-TPV', '-TAN-SIP']

_ap = {}
_ap['-TAN'] = {'name': 'scamp',
               'aprefix': 'pv1',
               'bprefix': 'pv2',
               'apprefix': 'pvi1',
               'bpprefix': 'pvi2'
               }

_ap['-TPV'] = _ap['-TAN']

_ap['-TAN-SIP'] = {'name': 'sip',
                   'aprefix': 'a',
                   'bprefix': 'b',
                   'apprefix': 'ap',
                   'bpprefix': 'bp'
                   }
_allowed_units = ['deg']

# same mapping for the inverse
smkeys = list(_scamp_map.keys())
for item in smkeys:
    newkey = item.replace('pv', 'pvi')
    _scamp_map[newkey] = _scamp_map[item]


class WCS:
    """
    A class to do WCS transformations.  Currently supports TAN projections
    for

        RA--TPV, DEC-TPV
        RA---TAN and DEC--TAN
        RA---TAN--SIP,DEC--TAN--SIP

    ctypes in degrees.  The first two are both actually TPV, but old versions
    of scamp wrote them simply as TAN.

    Usage:

    import wcsutil
    wcs = wcsutil.WCS(wcs_structure, longpole=180.0, latpole=90.0, theta0=90.0)

    The input is a wcs structure.  This could be a dictionary or numpy array
    that can be addressed like wcs['cunit1'] for example, or something that
    supports iteration like a fitsio header, or have an items() method such as
    for a pyfits header.  It is converted to a dictionary internally.  The
    structure is exactly that as would be written to a FITS header, so for
    example distortion fields are not converted to a matrix form, rather each
    field in the header gets a field in this structure.

    When there is a distortion model the inverse transformation is gotten
    by solving the for the roots of the transformation by default.  This
    is slow, so if you care about speed and not precision you can set
    find=False in sky2image() and it will use a polynomial fit to the inverse,
    which is calculated if not already in the header.

    The solve is done using scipy.optimize.fsolve

    Examples:
        # Use a fits header as initialization to a WCS class and convert
        # image (x,y) to equatorial longitude,latitude (ra,dec)
        import wcsutil
        import pyfits
        hdr=pyfits.getheader(fname)
        wcs = wcsutil.WCS(hdr)

        # convert x,y to ra,dec. x,y can be scalars or numpy arrays.
        # The returned ra,dec are always numpy arrays.
        ra,dec = wcs.image2sky(x,y)

        # the inverse.  When there is a distortion model prosent, by default
        # it finds the root of the forward transform, which is most accurate
        # way to do the inversion.  Send find=False to attempt to use an
        # inverse polynomial.

        x,y = wcs.sky2image(ra,dec)

    """
    def __init__(self, wcs, longpole=180.0, latpole=90.0, theta0=90.0):

        # Convert to internal dictionary and set some attributes of this
        # instance
        self.wcs = self.ConvertWCS(wcs)
        self._set_naxis()

        # Set these as attributes, either from above keywords or from the
        # wcs header
        self.SetAngles(longpole, latpole, theta0)

        # Now set a bunch more instance attributes from the wcs in a form
        # that is easier to work with
        self.ExtractFromWCS()

        # for finding the inverse trans
        self.lonlat_answer = numpy.zeros(2, dtype='f8')
        self.xyguess = numpy.zeros(2, dtype='f8')

    def __repr__(self):
        import pprint
        return pprint.pformat(self.wcs)

    def __getitem__(self, key):
        return self.wcs[key]

    def __setitem__(self, key, val):
        self.wcs[key] = val

    def keys(self):
        return list(self.wcs.keys())

    def get_naxis(self):
        """
        get [nx,ny], properly accounting for compressed data that
        use znaxis*

        returns
        -------
        [nx,ny] as an array
        """
        return self.naxis.copy()

    def get_jacobian(self, x, y, distort=True, step=1.0):
        """
        Get the elementes of the jacobian matrix at the specified locations
        This method currently assumes the system is ra,dec

        parameters
        ----------
        x,y: scalars or arrays
            x and y coords in the image
        distort:  bool, optional
            Use the distortion model if present.  Default is True
        step: float
            Step used for central difference formula, in pixels.  Default is
            1.0 pixels.

        returns
        -------
        jacobian elements: tuple of arrays
            dra_dx, dra_dy, ddec_dx, ddec_dy

        method
        ------
        Finite difference
        """

        fac = 1.0/(2*step)

        _, dec = self.image2sky(x, y, distort=distort)

        xp = x + step
        xm = x - step
        yp = y + step
        ym = y - step

        ra_p0, dec_p0 = self.image2sky(xp, y, distort=distort)
        ra_m0, dec_m0 = self.image2sky(xm, y, distort=distort)

        ra_0p, dec_0p = self.image2sky(x, yp, distort=distort)
        ra_0m, dec_0m = self.image2sky(x, ym, distort=distort)

        # in arcsec/pixel
        dra_dx = fac*3600.0*(ra_p0-ra_m0)
        dra_dy = fac*3600.0*(ra_0p-ra_0m)
        ddec_dx = fac*3600.0*(dec_p0-dec_m0)
        ddec_dy = fac*3600.0*(dec_0p-dec_0m)

        # need to scale dra b -cos(dec), minus sign since ra increases
        # to the left
        cosdec = -numpy.cos(dec*d2r)
        dra_dx *= cosdec
        dra_dy *= cosdec

        return dra_dx, dra_dy, ddec_dx, ddec_dy

    def image2sky(self, x, y, distort=True):
        """
        Convert between image x,y and sky coordinates lon,lat e.g. ra,dec.

        parameters
        ----------
        x,y: scalars or arrays
            x and y coords in the image
        distort:  bool, optional
            Use the distortion model if present.  Default is True

        returned values
        ---------------
        longitude,latitude:  tupple of arrays
            Probably ra,dec.  Will have the same shape as x,y

        examples
        --------
        import wcsutil
        import fitsio
        hdr=fitsio.read_header(fname)
        wcs = wcsutil.WCS(hdr)
        ra,dec = wcs.image2sky(x,y)
        """

        arescalar = isscalar(x)
        x = numpy.array(x, dtype='f8', copy=False)
        y = numpy.array(y, dtype='f8', copy=False)

        xdiff = x - self.crpix[0]
        ydiff = y - self.crpix[1]

        p = self.projection.upper()
        if p in ['-TAN', '-TPV']:
            u, v = self.ApplyCDMatrix(xdiff, ydiff)
            if distort and self.distort['name'] != 'none':
                # Assuming PV distortions
                u, v = self.Distort(u, v)

        elif p == '-TAN-SIP':      # pragma: no cover
            # this is broken as u and v are only defined if there is a distortion
            if distort and self.distort['name'] != 'none':
                u, v = self.Distort(xdiff, ydiff)
            u, v = self.ApplyCDMatrix(u, v)
        else:                    # pragma: no cover
            raise ValueError(f"projection '{p}' not supported")

        longitude, latitude = self.image2sph(u, v)
        if arescalar:
            longitude, latitude = longitude[0], latitude[0]

        return longitude, latitude

    def sky2image(self, lon, lat, distort=True, find=True, xtol=DEFTOL):
        """
        Usage:
            x,y=sky2image(longitude, latitude, distort=True, find=True)

        Purpose:
            Convert between sky (lon,lat) and image coordinates (x,y)

        Inputs:
            longitude,latitude:  Probably ra,dec. Can be arrays.
        Optional Inputs:
            distort:  Use the distortion model if present.  Default is True
            find: When the distortion model is present, simply find the
                roots of the polynomial rather than using an inverse
                polynomial.  This is more accurate but slower. Default True.
            xtol: tolerance to use when root finding with find=True Default is 1e-8.
        Outputs:
            x,y: x and y coords in the image.  Will have the same shape as
                lon,lat
        Example:
            import wcsutil
            import pyfits
            hdr=pyfits.getheader(fname)
            wcs = wcsutil.WCS(hdr)
            x,y = wcs.image2sky(ra,dec)
        """
        longitude = numpy.asarray(lon, dtype='f8')
        latitude = numpy.asarray(lat, dtype='f8')

        # Only do this if there is distortion
        if find and self.distort['name'] != 'none':
            x, y = self._findxy(longitude, latitude, xtol=xtol)
        else:
            u, v = self.sph2image(longitude, latitude)

            p = self.projection.upper()
            if p in ['-TAN', '-TPV']:
                if distort and self.distort['name'] != 'none':
                    u, v = self.Distort(u, v, inverse=True)
                xdiff, ydiff = self.ApplyCDMatrix(u, v, inverse=True)

            elif p == '-TAN-SIP':      # pragma: no cover
                u, v = self.ApplyCDMatrix(u, v, inverse=True)
                if distort and self.distort['name'] != 'none':
                    xdiff, ydiff = self.Distort(u, v, inverse=True)
                else:
                    xdiff, ydiff = u, v

            else:            # pragma: no cover
                raise ValueError(f"projection '{p}' not supported")

            x = xdiff + self.crpix[0]
            y = ydiff + self.crpix[1]

        return x, y

    def ExtractProjection(self, wcs):
        projection = wcs['ctype1'][4:].strip().upper()
        if projection not in _allowed_projections:
            err = ("Projection type {} unsupported.  Only [{}] projections currently supported")
            err = err.format(projection, ', '.join(_allowed_projections))
            raise ValueError(err)

        return projection

    def ApplyCDMatrix(self, x, y, inverse=False):
        if not inverse:
            cd = self.cd
            xp = cd[0, 0] * x + cd[0, 1] * y
            yp = cd[1, 0] * x + cd[1, 1] * y
        else:
            cdinv = self.cdinv
            xp = cdinv[0, 0] * x + cdinv[0, 1] * y
            yp = cdinv[1, 0] * x + cdinv[1, 1] * y

        return xp, yp

    def image2sph(self, x, y):
        """
        Convert x,y projected coordinates to spherical coordinates
        Currently only supports tangent plane projections.
        The conventions assumed are that of the WCS
        Works in the native system currently
        """
        if x.size != y.size:
            raise ValueError('x and y must be the same size')

        latitude = numpy.zeros_like(x) + math.pi/2

        # radius in radians
        r = numpy.sqrt(x**2 + y**2) * math.pi / 180.0

        scalar = isscalar(r)

        # not sure why this is being tested. r is always > 0
        if scalar:
            if r > 0:
                latitude = numpy.arctan(1.0 / r)
        else:
            w, = numpy.where(r > 0)
            if w.size > 0:
                latitude[w] = numpy.arctan(1.0 / r[w])

        longitude = numpy.arctan2(x, -y)

        longitude *= r2d
        latitude *= r2d

        longitude, latitude = self.Rotate(longitude, latitude, reverse=True)
        # Make sure the result runs from 0 to 360
        if scalar:
            if longitude < 0.0:
                longitude += 360.0

            if longitude >= 360.0:
                longitude -= 360.0

        else:
            w, = numpy.where(longitude < 0.0)
            if w.size > 0:
                longitude[w] += 360.0
            w, = numpy.where(longitude >= 360.0)  # pretty sure this can never happen
            if w.size > 0:      # pragma: no cover
                longitude[w] -= 360.0

        return longitude, latitude

    def sph2image(self, longitude, latitude):
        """
        Must be a tangent plane projection
        """
        longitude, latitude = self.Rotate(longitude, latitude)
        longitude *= d2r
        latitude *= d2r

        if longitude.size != latitude.size:    # pragma: no cover
            raise ValueError('long,lat must be the same size')

        x = numpy.zeros_like(longitude)
        y = numpy.zeros_like(longitude)

        if isscalar(longitude):
            if latitude > 0.0:
                rdiv = r2d / numpy.tan(latitude)
                x = rdiv*numpy.sin(longitude)
                y = -rdiv*numpy.cos(longitude)
        else:
            w, = numpy.where(latitude > 0.0)
            if w.size > 0:
                rdiv = r2d / numpy.tan(latitude[w])
                x[w] = rdiv * numpy.sin(longitude[w])
                y[w] = -rdiv * numpy.cos(longitude[w])

        return x, y

    def Rotate(self, lon, lat, reverse=False, origin=False):

        longitude = numpy.array(lon, ndmin=1, dtype='f8') * d2r
        latitude = numpy.array(lat, ndmin=1, dtype='f8') * d2r

        r = self.rotation_matrix
        if reverse:
            r = r.transpose()

        return self._rotate(longitude, latitude, r)

    def CreateRotationMatrix(self):
        # If Theta0 = 90 then CRVAL gives the coordinates of the origin in the
        # native system.   This must be converted (using Eq. 7 in Greisen &
        # Calabretta with theta0 = 0) to give the coordinates of the North
        # pole (longitude_p, latitude_p)

        # Longpole is the longitude in the native system of the North Pole in
        # the standard system (default = 180 degrees).
        sp = math.sin(self.longpole * d2r)
        cp = math.cos(self.longpole * d2r)

        sa = math.sin(self.native_longpole)
        ca = math.cos(self.native_longpole)
        sd = math.sin(self.native_latpole)
        cd = math.cos(self.native_latpole)

        # calculate rotation matrix

        # IDL array construction is transposed compared to python apparently
        # So this is reversed from the idl routines
        r = numpy.array([[-sa * sp - ca * cp * sd, sa * cp - ca * sp * sd, ca * cd],
                         [ca * sp - sa * cp * sd, -ca * cp - sa * sp * sd, sa * cd],
                         [cp * cd, sp * cd, sd]],
                        dtype='f8')

        return r

    def _rotate(self, longitude, latitude, r):
        """
        Apply a rotation matrix to the input longitude and latitude
        inputs must be numpy arrays
        """
        lat = numpy.cos(latitude) * numpy.cos(longitude)
        m = numpy.cos(latitude) * numpy.sin(longitude)
        n = numpy.sin(latitude)

        # find solution to the system of equations and put it in b
        # Can't use matrix notation in case l,m,n are rrays

        b0 = r[0, 0] * lat + r[1, 0] * m + r[2, 0] * n
        b1 = r[0, 1] * lat + r[1, 1] * m + r[2, 1] * n
        b2 = r[0, 2] * lat + r[1, 2] * m + r[2, 2] * n

        # Account for possible roundoff
        b2 = numpy.clip(b2, -1.0, 1.0)

        lat_new = numpy.arctan2(b2, numpy.sqrt(b0 * b0 + b1 * b1)) * r2d
        lon_new = numpy.arctan2(b1, b0) * r2d

        # there are no unittests so added this to make sure the new version works ok
        # if False:
        #    lat_new_old = numpy.arcsin(b2)*r2d
        #    assert numpy.allclose(lat_new_old,lat_new),"New WCS arctan function not working!"

        return lon_new, lat_new

    def _lonlatdiff(self, xy):
        x = numpy.array(xy[0])
        y = numpy.array(xy[1])
        lon, lat = self.image2sky(x, y)
        lonlat = numpy.zeros(2)
        lonlat[0] = lon
        lonlat[1] = lat
        diff = lonlat - self.lonlat_answer
        return diff

    def _fsolve_xy(self, xyguess, xtol=DEFTOL):
        xy = scipy.optimize.fsolve(self._lonlatdiff, xyguess, xtol=xtol)
        return xy

    def _lmfind_xy(self, xyguess):
        lm_tup = leastsq(self._lonlatdiff, xyguess, full_output=1)
        xy, _, _, errmsg, ier = lm_tup
        if ier > 4:
            raise RuntimeError("failed to find inverse transform: '%s'" % errmsg)
        return xy

    def _findxy(self, lon, lat, xtol=DEFTOL):
        """
        This is the simplest way to do the inverse of the (x,y)->(lon,lat)
        transformation when there are distortions.  Simply find the x,y
        that give the input lon,lat from the actual distortion function.

        Uses scipy.optimize.fsolve to find the roots of the transformation
        """

        if lon.size != lat.size:
            raise ValueError('lon and lat must be same size')

        if isscalar(lon):
            x, y = self._findxy_one(lon, lat, xtol=xtol)
        else:
            x = numpy.zeros_like(lon)
            y = numpy.zeros_like(lon)

            for i in range(lon.size):
                x[i], y[i] = self._findxy_one(lon[i], lat[i], xtol=xtol)

        return x, y

    def _findxy_one(self, lon, lat, xtol=DEFTOL):
        """
        This is the simplest way to do the inverse of the (x,y)->(lon,lat)
        transformation when there are distortions.  Simply find the x,y
        that give the input lon,lat from the actual distortion function.

        Uses scipy.optimize.fsolve to find the roots of the transformation
        """
        self.lonlat_answer[0] = lon
        self.lonlat_answer[1] = lat

        xyguess = self.xyguess

        # Use inversion without distortion as our guess
        xyguess[0], xyguess[1] = self.sky2image(lon, lat, find=False, distort=False)
        xy = self._fsolve_xy(xyguess, xtol=xtol)
        # print 'using lm'
        # xy = self._lmfind_xy(xyguess)
        x, y = xy[0], xy[1]

        return x, y

    def Distort(self, x, y, inverse=False):
        """
        Apply a distortion map to the data.  This follows the SIP convention,
        but if the scamp PV coefficients were found by the ConvertWCS code
        they are converted to the SIP convention.  The only difference is
        the order of operations:  for image to sky PV distortions come after
        the application of the CD matrix as opposed to SIP.

        """
        # Sometimes there is no distortion model present
        if self.distort is None or self.distort['name'] == 'none':
            # return copies
            return x*1.0, y*1.0

        if x.size != y.size:
            raise ValueError('x must be same size as y')

        if inverse:
            a = self.distort['ap']
            b = self.distort['bp']
        else:
            a = self.distort['a']
            b = self.distort['b']

        if self.distort['name'] == 'scamp':
            xp = 0 * x
            yp = 0 * y
        elif self.distort['name'] == 'sip':      # pragma: no cover
            xp = x * 1.0
            yp = y * 1.0
        else:      # pragma: no cover
            raise ValueError(f"Unsupported distortion model '{self.distort['name']}'")

        xp += Apply2DPolynomial(a, x, y)
        yp += Apply2DPolynomial(b, x, y)

        return xp, yp

    def _compare_inversion(self, x, y, xback, yback,
                           verbose=False, doplot=False, units=''):
        # Get rms differences
        t = (xback - x) ** 2 + (yback - y) ** 2
        rms = numpy.sqrt(t.sum() / t.size)
        if verbose:      # pragma: no cover
            mess = 'rms error'
            if units != '':
                mess += '('+units+')'
            mess += ':'
            sys.stdout.write(f'{mess} {rms}\n')
        if doplot:      # pragma: no cover
            import pylab
            pylab.clf()
            pylab.hist(x-xback, 50, edgecolor='black', fill=False)
            pylab.hist(y-yback, 50, edgecolor='red', fill=False)
            pylab.show()
        return rms

    def InvertDistortion(self, fac=5, order_increase=1,
                         verbose=False, doplot=False):
        if self.distort['name'] == 'scamp':
            return self.InvertPVDistortion(fac=fac,
                                           order_increase=order_increase,
                                           verbose=verbose,
                                           doplot=doplot)
        if self.distort['name'] == 'sip':      # pragma: no cover
            return self.InvertSipDistortion(fac=fac,
                                            order_increase=order_increase,
                                            verbose=verbose,
                                            doplot=doplot)

        raise ValueError('Can only invert scamp and sip distortions')      # pragma: no cover

    def InvertPVDistortion(self, fac=5, order_increase=1,
                           verbose=False, doplot=False):
        """
        Invert the distortion model.  Must contain a,b matrices
        """
        # Order of polynomial
        sx, _ = self.distort['a'].shape
        porder = sx - 1

        ng = 2 * (porder + 2)
        ng *= fac

        # Assuming 1 offset
        xrang = numpy.array([1.0, self.naxis[0]], dtype='f8') - self.crpix[0]
        yrang = numpy.array([1.0, self.naxis[1]], dtype='f8') - self.crpix[1]

        xdiff, ydiff = make_xy_grid(ng, xrang, yrang)

        # same to here
        u, v = self.ApplyCDMatrix(xdiff, ydiff)

        # This is what we will invert
        # up,vp = self.Distort(u,v)
        up = Apply2DPolynomial(self.distort['a'], u, v)
        vp = Apply2DPolynomial(self.distort['b'], u, v)
        # Find polynomial from up,vp to u,v
        ainv, binv = Invert2DPolynomial(up, vp, u, v, porder + order_increase)
        self.distort['ap'] = ainv
        self.distort['bp'] = binv

        # newu, newv = self.Distort(up, vp, inverse=True)
        newu = Apply2DPolynomial(ainv, up, vp)
        newv = Apply2DPolynomial(binv, up, vp)
        ufrac = (u - newu) / u
        vfrac = (v - newv) / v
        if verbose:      # pragma: no cover
            sys.stdout.write('\ntesting inverse now:\n')
            sys.stdout.write(f'\n  ufrac={ufrac}\n')
            sys.stdout.write(f'  vfrac={vfrac}\n')
            sys.stdout.write(f'\n  median ufrac={numpy.median(ufrac)}\n')
            sys.stdout.write(f'  median vfrac= {numpy.median(vfrac)}\n\n')

        _ = self._compare_inversion(u, v, newu, newv,
                                    verbose=verbose, doplot=doplot)

        x = xdiff + self.crpix[0]
        y = ydiff + self.crpix[1]
        lon, lat = self.image2sky(x, y)
        xback, yback = self.sky2image(lon, lat, find=False)

        rms = self._compare_inversion(x, y, xback, yback, verbose=verbose,
                                      doplot=doplot, units='pixels')
        return rms

    def InvertSipDistortion(self, fac=5, verbose=False, doplot=False, order_increase=1):      # pragma: no cover
        """
        Invert the distortion model.  Must contain a,b matrices
        """
        # Order of polynomial
        sx, _ = self.distort['a'].shape
        porder = sx - 1

        ng = 2 * (porder + 2)
        ng *= fac

        xrang = numpy.array([1.0, self.naxis[0]])
        yrang = numpy.array([1.0, self.naxis[1]])

        x, y = make_xy_grid(ng, xrang, yrang)

        # Use distortion for getting sky coords
        lon, lat = self.image2sky(x, y)
        # Don't use distortion to get back image coords.  We will use
        # the difference to fit for new coefficients.
        xback, yback = self.sky2image(lon, lat, distort=False, find=False)
        self._compare_inversion(x, y, xback, yback,
                                verbose=verbose, doplot=doplot)

        xdiff = xback - self.crpix[0]
        ydiff = yback - self.crpix[1]

        constant = False
        ainv, binv = Invert2DPolynomial(xdiff, ydiff, x - xback, y - yback,
                                        porder + order_increase,
                                        constant=constant)

        if 'ap' in self.distort:
            xback2, yback2 = self.sky2image(lon, lat, find=False)
            rms = self._compare_inversion(x, y, xback2, yback2,
                                          verbose=verbose,
                                          doplot=doplot, units='pixels')

        self.distort['ap'] = ainv
        self.distort['bp'] = binv

        xback2, yback2 = self.sky2image(lon, lat, find=False)

        rms = self._compare_inversion(x, y, xback2, yback2,
                                      verbose=verbose, doplot=doplot)
        return rms

    def GetPole(self):

        longitude_0 = self.wcs['crval1'] * d2r
        latitude_0 = self.wcs['crval2'] * d2r

        if self.theta0 == 90.0:
            return longitude_0, latitude_0

        # Longpole is the longitude in the native system of the North Pole
        # in the standard system (default = 180 degrees).
        phi_p = self.longpole * d2r
        sp = math.sin(phi_p)
        cp = math.cos(phi_p)
        sd = math.sin(latitude_0)
        cd = math.cos(latitude_0)
        tand = math.tan(latitude_0)

        if self.theta0 == 0.0:
            if latitude_0 == 0 and self.longpole == 90.0:
                latitude_p = self.latpole
            else:
                latitude_p = math.acos(sd / cp)

            if self.latpole != 90.0:
                if math.fabs(self.latpole + latitude_p) < math.fabs(self.latpole - latitude_p):
                    latitude_p = - latitude_p
            if (self.longpole == 180.0) or (cd == 0.0):
                longitude_p = longitude_0
            else:
                longitude_p = longitude_0 - math.atan2(sp / cd, -math.tan(latitude_p) * tand)
        else:
            ctheta = math.cos(self.theta0 * d2r)
            stheta = math.sin(self.theta0 * d2r)

            term1 = math.atan2(stheta, ctheta * cp)
            term2 = math.acos(sd / (math.sqrt(1.0 - ctheta * ctheta * sp * sp)))

            # not sure if term2 can actually be 0.0 as it takes a very specialized set of circumstances
            # (to the 11+ significant figure) Also this will throw an error, because longitude_p
            # is never defined by this branch
            if term2 == 0.0:      # pragma: no cover
                latitude_p = term1
            else:
                latitude_p1 = math.fabs((term1 + term2) * r2d)
                latitude_p2 = math.fabs((term1 - term2) * r2d)

                if latitude_p1 > 90.0 and latitude_p2 > 90.0:
                    raise ValueError('No valid solution')
                if latitude_p1 < 90.0 < latitude_p2:
                    latitude_p = term1 + term2
                elif latitude_p2 < 90.0 < latitude_p1:
                    latitude_p = term1 - term2
                else:
                    # Two valid solutions
                    latitude_p1 = (term1 + term2) * r2d
                    latitude_p2 = (term1 - term2) * r2d
                    if math.fabs(self.latpole-latitude_p1) < \
                       math.fabs(self.latpole-latitude_p2):
                        latitude_p = term1+term2
                    else:
                        latitude_p = term1-term2

                # changed this to be a less than because while cos(PI/2) is defined as 0.0 in
                # reality the floating point accuracy makes it 6.12e-17. Using a limit of 1e-10
                # gives an accuracy of PI/2 to 8 decimal places
                # although this cannot really be excercised as term2 == 0 when latitude_0 approaches
                # PI/2
                # if cd == 0.0:
                if abs(cd) < 1e-10:     # pragma: no cover
                    longitude_p = longitude_0
                else:
                    sdelt = math.sin(latitude_p)
                    if sdelt == 1.0:
                        longitude_p = longitude_0 - phi_p - math.pi
                    else:
                        if sdelt == -1.0:
                            longitude_p = longitude_0 - phi_p
                        else:
                            sdp = math.sin(latitude_p)
                            cdp = math.cos(latitude_p)
                            longitude_p = longitude_0 - math.atan2((stheta - sdp * sd) / (cdp * cd), sp * ctheta / cd)
        return longitude_p, latitude_p
    # disableing coverage because this method is problematic
    # it wants a numpy array with fields, but then doen't care what those fields are
    # or a dict - which seems ok
    # or something else that is also a dict in structure, but not an actual dict (or subclass)
    # or yet something else that is also a dict in structure, but not an actual dict (or subclass)

    def ConvertWCS(self, wcs_in):     # pragma: no cover
        """
        Convert to a dictionary
        """

        self.wcs = None
        self.distort = {'name': 'none'}
        self.cd = None
        self.crpix = None
        self.crval = None
        self.projection = None

        # Convert the wcs to a local dictionary

        wcs = {}
        if isinstance(wcs_in, numpy.ndarray) or hasattr(wcs_in, 'dtype'):
            if wcs_in.dtype.fields is None:
                raise ValueError('wcs array must have fields')

            for f in wcs_in.dtype.fields:
                fl = f.lower()
                val = wcs_in[f]
                if val.ndim == 0:
                    wcs[fl] = val
                else:
                    # only scalars
                    wcs[fl] = val[0]

        elif isinstance(wcs_in, dict):
            wcs = wcs_in.copy()
        elif hasattr(wcs_in, '__iter__'):
            wcs = {}
            for k in wcs_in:
                if k is None:
                    continue
                wcs[k.lower()] = wcs_in[k]
        else:
            # Try to use the items() method to get what we want
            wcs = {}
            try:
                for k, v in wcs_in.items():
                    if k is None:
                        continue
                    wcs[k.lower()] = v
            except ValueError:
                raise ('Input wcs must be a numpy array '
                       'with fields or a dictionary or support '
                       'iteration or an items() method')
        return wcs

    def SetAngles(self, longpole, latpole, theta0):
        # These can get set if they were not in the WCS header
        if 'longpole' not in self.wcs:
            self.longpole = longpole
        else:
            self.longpole = self.wcs['longpole']

        if 'latpole' not in self.wcs:
            self.latpole = latpole
        else:
            self.latpole = self.wcs['latpole']

        if 'theta0' not in self.wcs:
            self.theta0 = theta0
        else:
            self.theta0 = self.wcs['theta0']

    # disabling coverage as this method is not well described (what form does wcs take?)
    def ExtractUnits(self, wcs):     # pragma: no cover

        if 'cunit1' in wcs:
            units = wcs['cunit1'].strip().lower()
            if units not in _allowed_units:
                err = 'Unsupported units {}.  Only [{}] supported'
                raise ValueError(err.format(units, ', '.join(_allowed_units)))
        else:
            units = None
        return units

    def ExtractDistortCoeffs(self, dname, wcs, prefix):
        if dname == 'scamp':
            return self.ExtractPVCoeffs(wcs, prefix)
        if dname == 'sip':      # pragma: no cover
            return self.ExtractSIPCoeffs(wcs, prefix)

    def ExtractPVCoeffs(self, wcs, prefix):
        order = _scamp_max_order
        dim = order + 1
        matrix = numpy.zeros((dim, dim), dtype='f8')
        count = 0
        for i in range(_scamp_max_ncoeff):
            if i not in _scamp_skip:
                key = prefix + '_' + str(i)
                if key in wcs:
                    indices = _scamp_map[key]
                    matrix[indices[0], indices[1]] = wcs[key]
                    count += 1
        return matrix, count, order

    def ExtractSIPCoeffs(self, wcs, prefix):      # pragma: no cover
        order = _dict_get(wcs, prefix + '_order')
        matrix = numpy.zeros((order + 1, order + 1), dtype='f8')
        count = 0
        for ix in range(order + 1):
            for iy in range(order + 1):
                key = prefix + '_' + str(ix) + '_' + str(iy)
                if key in wcs:
                    matrix[ix, iy] = wcs[key]
                    count += 1
        return matrix, count, order

    def ExtractDistortionModel(self):
        if self.projection not in _allowed_projections:
            raise ValueError(f"Projection must be on of {', '.join(_allowed_projections)} ")

        # look for forward coeffs first
        dinfo = _ap[self.projection]
        dname = dinfo['name']
        a, ca, aorder = self.ExtractDistortCoeffs(dname,
                                                  self.wcs,
                                                  dinfo['aprefix'])

        if ca != 0:
            self.distort['name'] = dname

            b, _, border = \
                self.ExtractDistortCoeffs(dname,
                                          self.wcs,
                                          dinfo['bprefix'])
            ap, cap, aporder = \
                self.ExtractDistortCoeffs(dname,
                                          self.wcs,
                                          dinfo['apprefix'])
            bp, cbp, bporder = \
                self.ExtractDistortCoeffs(dname,
                                          self.wcs,
                                          dinfo['bpprefix'])

            self.distort['a'] = a
            self.distort['a_order'] = aorder
            self.distort['b'] = b
            self.distort['b_order'] = border

            # these coeffs will be zeros if not found above
            self.distort['ap'] = ap
            self.distort['ap_order'] = aporder
            self.distort['bp'] = bp
            self.distort['bp_order'] = bporder

            # If inverse not there, calculate it
            if cap == 0 or cbp == 0:
                self.InvertDistortion()
                self.distort['ap_order'] = self.distort['a_order'] + 1
                self.distort['bp_order'] = self.distort['b_order'] + 1

    def ExtractFromWCS(self):

        # for easier notation
        wcs = self.wcs

        # set these to little arrays
        self.crpix = numpy.array([wcs['crpix1'],
                                  wcs['crpix2']], dtype='f8')
        self.crval = numpy.array([wcs['crval1'],
                                  wcs['crval2']], dtype='f8')
        self.ctype = numpy.array([wcs['ctype1'].strip().upper(),
                                  wcs['ctype2'].strip().upper()])

        # Get the projection from ctype
        self.projection = self.ExtractProjection(wcs)

        # Get units
        self.units = self.ExtractUnits(wcs)

        # CTYPE[0] - first four characters specify standard system
        #       ('RA--','GLON' or 'ELON' for right ascension, galactic
        #       longitude or ecliptic longitude respectively), second four
        #       letters specify the type of map projection (eg '-AIT' for
        #       Aitoff projection)
        # CTYPE[1] - first four characters specify standard system
        #       ('DEC-','GLAT' or 'ELAT' for declination, galactic latitude
        #       or ecliptic latitude respectively; these must match
        #       the appropriate system of ctype1), second four letters of
        #       ctype2 must match second four letters of ctype1.

        system1 = self.wcs['ctype1'][0:4]
        system2 = self.wcs['ctype2'][0:4]
        self.system = numpy.array([system1, system2], dtype='S4')

        # Add a 2x2 array for the cd matrix
        if 'cd1_1' in wcs:
            cd = numpy.zeros((2, 2), dtype='f8')
            cd[0, 0] = wcs['cd1_1']
            cd[0, 1] = wcs['cd1_2']
            cd[1, 0] = wcs['cd2_1']
            cd[1, 1] = wcs['cd2_2']
            self.cd = cd

            try:
                self.cdinv = numpy.linalg.inv(cd)
            except ValueError:
                raise ('Could not find inverse of CD matrix')
        # Get the poles for the inputs.  Assumes we already ran
        # SetAngles() before calling this method
        self.native_longpole, self.native_latpole = self.GetPole()

        # Create the rotation matrix for later.  Requires that the
        # native system be set up using GetPole()
        self.rotation_matrix = self.CreateRotationMatrix()

        # Extract the distortion model
        self.ExtractDistortionModel()

    def _set_naxis(self):
        wcs = self.wcs
        if 'znaxis1' in wcs:
            self.naxis = numpy.array([wcs['znaxis1'], wcs['znaxis2']])
        else:
            self.naxis = numpy.array([wcs['naxis1'], wcs['naxis2']])


def _dict_get(d, key, default=None):
    if key not in d:
        if default is not None:
            return default
        raise ValueError(f"key '{key}' must be present")
    return d[key]


def arrscl(arr, minval, maxval, arrmin=None, arrmax=None):
    # makes a copy either way (asarray would not if it was an array already)
    output = numpy.array(arr)

    if arrmin is None:
        arrmin = output.min()
    if arrmax is None:
        arrmax = output.max()

    if output.size == 1:
        return output

    if arrmin == arrmax:
        sys.stdout.write('arrmin must not equal arrmax\n')
        return None

    try:
        a = (maxval - minval) / (arrmax - arrmin)
        b = (arrmax*minval - arrmin*maxval) / (arrmax - arrmin)
    except ValueError:  # pragma: no cover
        sys.stdout.write(f"Error calculating a,b: {sys.exc_info()[0]} {sys.exc_info()[1]}\n")
        return None

    # in place
    numpy.multiply(output, a, output)
    numpy.add(output, b, output)

    return output


def Apply2DPolynomial(a, x, y):
    v = numpy.zeros_like(x)

    sx, sy = a.shape
    for ix in range(sx):
        for iy in range(sy):
            xpow = x ** ix
            ypow = y ** iy
            if a[ix, iy] != 0.0:
                addval = a[ix, iy] * xpow * ypow
                v += addval

    return v


def make_xy_grid(n, xrang, yrang):
    # Create a grid on input ranges
    rng = numpy.arange(n, dtype='f8')
    ones = numpy.ones(n, dtype='f8')

    x = arrscl(rng, xrang[0], xrang[1])
    y = arrscl(rng, yrang[0], yrang[1])

    x = numpy.outer(x, ones)
    y = numpy.outer(ones, y)
    x = x.flatten('F')
    y = y.flatten('F')

    return x, y


def make_amatrix(u, v, order, constant=True):
    # matrix for inversion.
    # coeffs_u = A^{-1} x = (a^Ta)^{-1} A^T x
    # coeffs_v = A^{-1} v

    # n = (order+1)*2
    # n = n*n
    n = u.size

    tshape = [(order + 1) * (order + 2) // 2 - 1, n]
    if constant:
        # Extra column with ones in it for the constant term
        tshape[0] += 1
        kstart = 1
    else:
        kstart = 0
    # amatrix = numpy.zeros( tshape )
    amatrix = numpy.ones(tshape)

    kk = kstart
    for order in range(1, order + 1):
        for jj in range(order + 1):
            amatrix[kk, :] = (u ** (order - jj)) * v ** jj
            kk += 1

    return amatrix


def invert_for_coeffs(amatrix, x, y, lsolve=True):
    # a^T a
    ata = numpy.inner(amatrix, amatrix)
    # a^T x
    atx = numpy.inner(amatrix, x)
    # a^T y
    aty = numpy.inner(amatrix, y)

    if lsolve:
        # More stable solver
        xcoeffs = numpy.linalg.solve(ata, atx)
        ycoeffs = numpy.linalg.solve(ata, aty)

    else:
        atainv = numpy.linalg.inv(ata)
        # atainv = numpy.linalg.pinv(ata)
        xcoeffs = numpy.inner(atainv, atx)
        ycoeffs = numpy.inner(atainv, aty)

    return xcoeffs, ycoeffs


def pack_coeffs(xcoeffs, ycoeffs, porder, constant=True):
    """
    pack coeffs into a matrix form
    """

    if constant:
        ostart = 0
    else:
        ostart = 1

    kk = 0
    shape = (porder + 1, porder + 1)
    ainv = numpy.zeros(shape)
    binv = numpy.zeros(shape)
    for order in range(ostart, porder + 1):
        for jj in range(order + 1):
            ainv[order - jj, jj] = xcoeffs[kk]
            binv[order - jj, jj] = ycoeffs[kk]
            kk += 1
    return ainv, binv


# Find the polynomial coeffs that take us from u,v to x,y
def Invert2DPolynomial(u, v, x, y, porder, pack=True, constant=True):
    # matrix for inversion.
    # coeffs_u = A^{-1} x = (A^TA)^{-1} A^T x
    # coeffs_v = A^{-1} v
    amatrix = make_amatrix(u, v, porder, constant=constant)

    # Now we know the inverse must equal x,y so we use that as the
    # constraint vector
    xcoeffs, ycoeffs = invert_for_coeffs(amatrix, x, y)

    if pack:
        # now pack the coefficients into a matrix
        ainv, binv = pack_coeffs(xcoeffs, ycoeffs, porder, constant=constant)
        return ainv, binv
    return xcoeffs, ycoeffs


def Ncoeff(order, constant=True):
    ncoeff = (order + 1) * (order + 2) // 2
    if not constant:
        ncoeff -= 1
    return ncoeff


def test_invert_2dpoly(porder, fac=5, constant=True, order_increase=0, inverse=False):   # pragma: no cover
    import pylab

    # total number of constraints should be at least equal to the
    # number of coeffs.  If constant term is included, ncoeff is
    #  (order+1)*(order+2)/2 < (order+2)^2/2 < (order+2)^2
    # So let's do a lot: 20*(order+2)^2

    if porder > 3:
        raise ValueError('Only testing up to order 3 right now')

    # in making the grid we will square this n
    n = 2 * (porder + 2)
    n *= fac

    cen = [500.0, 1000.0]
    u, v = make_xy_grid(n, [1.0, 1000.0], [1.0, 2000.0])
    u -= cen[0]
    v -= cen[1]

    if constant:
        x0 = 2.0
        y0 = 3.0
        start = 0
    else:
        start = 1
        x0 = 0.0
        y0 = 0.0

    ucoeffs_in = numpy.array(
        [x0, 0.1, 0.2, 0.05, 0.03, 0.04, 0.005, 0.004, 0.001, 0.0009], dtype='f8')
    vcoeffs_in = numpy.array(
        [y0, 0.3, 0.5, 0.06, 0.05, 0.06, 0.004, 0.008, 0.003, 0.002], dtype='f8')
    ucoeffs_in = numpy.array(
        [x0,
         1.0, 1.e-2,
         5.e-3, 3.e-3, 4.e-3,
         0.0, 0.0, 0.0, 0.0], dtype='f8')
    vcoeffs_in = numpy.array(
        [y0,
         1.0, 2.e-2,
         6.e-3, 5.5e-3, 4.e-3,
         0.0, 0.0, 0.0, 0.0], dtype='f8')

    # number to actuall use
    ncoeff = (porder + 1) * (porder + 2) // 2
    keep = numpy.arange(start, ncoeff)
    ucoeffs_in = ucoeffs_in[keep]
    vcoeffs_in = vcoeffs_in[keep]

    ainv, binv = pack_coeffs(ucoeffs_in, vcoeffs_in, porder, constant=constant)
    x = Apply2DPolynomial(ainv, u, v)
    y = Apply2DPolynomial(binv, u, v)

    pylab.clf()

    if not inverse:
        # get poly from u,v to x,y
        ucoeffs, vcoeffs = Invert2DPolynomial(u, v, x, y, porder, pack=False,
                                              constant=constant)
        ucoeffsp, vcoeffsp = Invert2DPolynomial(u, v, x, y, porder, pack=True,
                                                constant=constant)
        newx = Apply2DPolynomial(ucoeffsp, u, v)
        newy = Apply2DPolynomial(vcoeffsp, u, v)

        w, = numpy.where((numpy.abs(x) > 5) & (numpy.abs(y) > 5))
        xfrac = (x[w] - newx[w]) / x[w]
        yfrac = (y[w] - newy[w]) / y[w]

        sys.stdout.write(f'umax,umin= {u.max()},{u.min()} vmax,vmin = {v.max()},{v.min()}\n\n')
        sys.stdout.write(f'median(xfracerr)={numpy.median(xfrac)}\n')
        sys.stdout.write(f'median(abs(xfracerr))={numpy.median(numpy.abs(xfrac))}\n')
        sys.stdout.write(f'sdev(xfracerr)={xfrac.std()}\n')
        sys.stdout.write(f'median(yfracerr)={numpy.median(yfrac)}\n')
        sys.stdout.write(f'median(abs(yfracerr))={numpy.median(numpy.abs(yfrac))}\n')
        sys.stdout.write(f'sdev(yfracerr)={yfrac.std()}\n\n')
        pylab.subplot(2, 1, 1)
        pylab.hist(xfrac, 50)
        pylab.hist(yfrac, 50, edgecolor='red', fill=False)
        sys.stdout.write(f'ucoeffs_in ={ucoeffs_in}\n')
        sys.stdout.write(f'ucoeffs_found ={ucoeffs}\n')
        sys.stdout.write(f'vcoeffs_in ={vcoeffs_in}\n')
        sys.stdout.write(f'vcoeffs_found ={vcoeffs}\n')

    else:
        # Now test the inverse, from x,y to u,v
        sys.stdout.write('\nTesting inverse\n')
        xcoeffs, ycoeffs = Invert2DPolynomial(x, y, u, v, porder + order_increase, pack=False, constant=constant)
        xcoeffsp, ycoeffsp = Invert2DPolynomial(x, y, u, v, porder + order_increase, pack=True, constant=constant)
        newu = Apply2DPolynomial(xcoeffsp, x, y)
        newv = Apply2DPolynomial(ycoeffsp, x, y)

        sys.stdout.write(f'{u[0:25]}\n')
        sys.stdout.write(f'{newu[0:25]}\n')

        w, = numpy.where((numpy.abs(u) > 5) & (numpy.abs(v) > 5))
        ufrac = (u[w] - newu[w]) / u[w]
        vfrac = (v[w] - newv[w]) / v[w]

        sys.stdout.write(f'xcoeffs{xcoeffs}\n')
        sys.stdout.write(f'ycoeffs{ycoeffs}\n\n')
        sys.stdout.write(f'median(ufracerr){numpy.median(ufrac)}\n')
        sys.stdout.write(f'median(abs(ufracerr)){numpy.median(numpy.abs(ufrac))}\n')
        sys.stdout.write(f'sdev(ufracerr){ufrac.std()}\n')
        sys.stdout.write(f'median(vfracerr){numpy.median(vfrac)}\n')
        sys.stdout.write(f'median(abs(vfracerr)){numpy.median(numpy.abs(vfrac))}\n')
        sys.stdout.write(f'sdev(vfracerr){vfrac.std()}\n\n')
        # pylab.subplot(2,1,2)
        pylab.hist(ufrac, 50)
        pylab.hist(vfrac, 50, edgecolor='red', fill=False)

    pylab.show()
