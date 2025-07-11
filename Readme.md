# des_cutter

A python module to make FITS files cutouts/thumbnails

Description
-----------

Set of libraries and scripts to create thumbnails/cutouts from DES images and color-composed RGB images.

Features
--------
- It reads the inputs positions (RA,DEC) in decimals form a CSV file, with optional XSIZE,YSIZE in arc-minutes.
- Can be run single or multi-threaded
- Uses fitsio to open/write files
- Used stiff to make color images
- Can choose bands (--bands option) to cut from.

Examples
--------
To get the thumbnails for the positions in the file: inputfile_radec.csv in multi-process mode

```
   des_cutter inputfile_radec.csv --xsize 1.5 --ysize 1.5  --MP
```
