-- DECADE queries using

-- COADD images
select c.FILENAME, c.TILENAME, c.BAND, f.PATH, f.COMPRESSION,
       t.CROSSRA0, t.RACMIN, t.RACMAX, t.DECCMIN, t.DECCMAX,
       t.RA_CENT, t.DEC_CENT, t.RA_SIZE, t.DEC_SIZE,
       t.RAC1, t.RAC2, t.RAC3, t.RAC4, t.DECC1, t.DECC2, t.DECC3, t.DECC4
 from COADD c, PROCTAG p, FILE_ARCHIVE_INFO f, COADDTILE_GEOM t
  where p.TAG = 'DR3_COADD'
	  and t.tilename = c.TILENAME
    and c.FILETYPE = 'coadd'
    and p.PFW_ATTEMPT_ID = c.PFW_ATTEMPT_ID
    and f.FILENAME = c.FILENAME;

-- COADD catalogs
select c.FILENAME, c.TILENAME, c.BAND, c.FILETYPE, f.PATH
  from CATALOG c, PROCTAG p, FILE_ARCHIVE_INFO f
   where p.TAG = 'DR3_COADD'
    and c.FILENAME = f.FILENAME
    and c.FILETYPE='coadd_cat'
    and p.PFW_ATTEMPT_ID = c.PFW_ATTEMPT_ID and ROWNUM < 50
    ORDER BY c.FILENAME;

-- FINALCUT IMAGES
select i.FILENAME, f.PATH, f.COMPRESSION, i.BAND, i.EXPTIME, i.AIRMASS, i.FWHM, i.NITE, i.EXPNUM, i.CCDNUM, i.PFW_ATTEMPT_ID,
       e.DATE_OBS, e.MJD_OBS,
       i.CROSSRA0, i.RACMIN, i.RACMAX, i.DECCMIN, i.DECCMAX,
       i.RA_CENT, i.DEC_CENT,
       i.RAC1, i.RAC2, i.RAC3, i.RAC4, i.DECC1, i.DECC2, i.DECC3, i.DECC4,
       (case when i.CROSSRA0='Y' THEN abs(i.RACMAX - (i.RACMIN-360)) ELSE abs(i.RACMAX - i.RACMIN) END) as RA_SIZE,
       abs(i.DECCMAX - i.DECCMIN) as DEC_SIZE
 from IMAGE i, EXPOSURE e, FILE_ARCHIVE_INFO f, PROCTAG p
  where p.TAG = 'DR3_FINALCUT'
   and p.pfw_attempt_id = i.pfw_attempt_id
   and i.EXPNUM=e.EXPNUM
   and i.FILENAME=f.FILENAME
   and i.FILETYPE='red_immask';

-- FINALCUT CATALOGS
select c.FILENAME, f.PATH, c.FILETYPE, c.BAND, c.CCDNUM, c.PFW_ATTEMPT_ID
 from CATALOG c, FILE_ARCHIVE_INFO f, PROCTAG p
  where p.TAG = 'DR3_FINALCUT'
    and p.pfw_attempt_id = i.pfw_attempt_id
    and f.FILENAME=c.FILENAME
    and c.FILETYPE='cat_finalcut';


select i.FILENAME, i.EXPNUM, i.CCDNUM, i.BAND, i.PFW_ATTEMPT_ID, p.TAG
 from IMAGE i
 join PROCTAG p on p.PFW_ATTEMPT_ID = i.PFW_ATTEMPT_ID
 where p.TAG = 'DR3_FINALCUT'
 and i.BAND='r'
 and i.FILETYPE = 'red_immask' and ROWNUM < 100;
