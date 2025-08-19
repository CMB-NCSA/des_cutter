--- *** FINALCUT TABLES *** ---

--- Creation of table on dessci for Y6A2_FINALCUT_FILEPATH
--- build table with info for finalcut files
-- DROP TABLE felipe.Y6A2_FINALCUT_FILEPATH;
create table Y6A2_FINALCUT_FILEPATH as
select i.FILENAME, f.PATH, f.COMPRESSION, i.BAND, i.EXPTIME, i.AIRMASS, i.FWHM, i.NITE, i.EXPNUM, i.CCDNUM, i.PFW_ATTEMPT_ID,
       e.DATE_OBS, e.MJD_OBS,
       i.CROSSRA0, i.RACMIN, i.RACMAX, i.DECCMIN, i.DECCMAX,
       i.RA_CENT, i.DEC_CENT,
       i.RAC1, i.RAC2, i.RAC3, i.RAC4, i.DECC1, i.DECC2, i.DECC3, i.DECC4,
       (case when i.CROSSRA0='Y' THEN abs(i.RACMAX - (i.RACMIN-360)) ELSE abs(i.RACMAX - i.RACMIN) END) as RA_SIZE,
       abs(i.DECCMAX - i.DECCMIN) as DEC_SIZE
 from Y6A2_IMAGE i, Y6A2_EXPOSURE e, Y6A2_FILE_ARCHIVE_INFO f
 where  i.EXPNUM=e.EXPNUM
        and i.FILENAME=f.FILENAME
        and i.FILETYPE='red_immask';

-- SELECT COUNT(*) FROM Y6A2_FINALCUT_FILEPATH;
--   7954954

-- drop table felipe.Y6A2_FINALCUT_CATALOG_FILEPATH
create table Y6A2_FINALCUT_CATALOG_FILEPATH as
select c.FILENAME, f.PATH, c.FILETYPE, c.BAND, c.CCDNUM, c.PFW_ATTEMPT_ID
 from des_admin.Y6A2_CATALOG c, des_admin.Y6A2_FILE_ARCHIVE_INFO f
  where f.FILENAME=c.FILENAME
    and c.FILETYPE='cat_finalcut';

-- SELECT COUNT(*) FROM Y6A2_FINALCUT_CATALOG_FILEPATH;
--       7954954

---- *** COADD TABLES *** ---
-- To create a new table on dessci for Y6A2 called felipe.Y6A2_COADD_FILEPATH:
-- Now with corners information
-- DROP TABLE felipe.Y6A2_COADD_FILEPATH
create table Y6A2_COADD_FILEPATH as
select c.FILENAME, c.TILENAME, c.BAND, c.FILETYPE, f.PATH, f.COMPRESSION,
       t.CROSSRA0, t.RACMIN, t.RACMAX, t.DECCMIN, t.DECCMAX,
       t.RA_CENT, t.DEC_CENT, t.RA_SIZE, t.DEC_SIZE,
       t.RAC1, t.RAC2, t.RAC3, t.RAC4, t.DECC1, t.DECC2, t.DECC3, t.DECC4
 from des_admin.Y6A2_COADD c, des_admin.Y6A2_FILE_ARCHIVE_INFO f, des_admin.Y6A1_COADDTILE_GEOM t
  where f.FILENAME=c.FILENAME
      and c.TILENAME = t.TILENAME
      and c.FILETYPE='coadd';

-- SELECT COUNT(*) FROM Y6A2_COADD_FILEPATH;
-- COUNT(*)
----------
-- 50845

create table Y6A2_COADD_CATALOG_FILEPATH as
select c.FILENAME, c.TILENAME, c.BAND, c.FILETYPE, f.PATH, d.FILESIZE --, d.MD5SUM, d.COMPRESSION
 from des_admin.Y6A2_CATALOG c, des_admin.Y6A2_FILE_ARCHIVE_INFO f -- , des_admin.Y6A2_DESFILE_COLUMNS d
  where c.FILENAME = f.FILENAME
    and d.FILENAME = c.FILENAME
    and c.FILETYPE='coadd_cat';

-- SELECT COUNT(*) FROM Y6A2_COADD_CATALOG_FILEPATH;
-- 50845



--------------------------------------------------
-- example of search near ra, dec
select i.FILENAME, i.BAND, i.FILETYPE, i.EXPTIME, i.NITE, i.EXPNUM, e.DATE_OBS, e.MJD_OBS from Y6A2_IMAGE i, Y6A2_EXPOSURE e
  where ((i.CROSSRA0='N' AND (0.29782658 BETWEEN i.RACMIN and i.RACMAX) AND (0.029086056 BETWEEN i.DECCMIN and i.DECCMAX)) OR
        (i.CROSSRA0='Y' AND (0.29782658 BETWEEN i.RACMIN-360 and i.RACMAX) AND (0.029086056 BETWEEN i.DECCMIN and i.DECCMAX)))
        and i.EXPNUM=e.EXPNUM
        and i.BAND='r'
        and i.FILETYPE='red_immask';

-- example test search near ra, dec for finalcut
SET TIMING ON;
select i.FILENAME, i.BAND, i.EXPTIME, i.NITE, i.EXPNUM, i.DATE_OBS, i.MJD_OBS from felipe.Y6A2_FINALCUT_FILEPATH i
  where ((i.CROSSRA0='N' AND (0.29782658 BETWEEN i.RACMIN and i.RACMAX) AND (0.029086056 BETWEEN i.DECCMIN and i.DECCMAX)) OR
  (i.CROSSRA0='Y' AND (0.29782658 BETWEEN i.RACMIN-360 and i.RACMAX) AND (0.029086056 BETWEEN i.DECCMIN and i.DECCMAX)))
  and i.DATE_OBS between '2013-10-25T00:05:49' and '2014-12-25T00:05:49'
  and i.BAND='r' order by i.EXPNUM;
SET TIMING OFF;


SELECT segment_type, SUM(bytes) / 1024 / 1024 AS size_mb FROM user_segments WHERE segment_name = 'Y6A2_FINALCUT_FILEPATH' OR segment_name IN (SELECT index_name FROM user_indexes WHERE table_name = 'Y6A2_FINALCUT_FILEPATH')
GROUP BY segment_type;
