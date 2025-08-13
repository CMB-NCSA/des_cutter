-- DECADE example
-- create table DR3_COADD_FILEPATH as

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
