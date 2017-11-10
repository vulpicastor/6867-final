#!/usr/bin/python3

from astropy import table
from astropy.io import fits, ascii
import logging
import numpy as np
from os import path
import sys

import injection

BOUND_TOL = 1. / 24  # day; this is equivalent to two long cadences.

def dictify(fits_header):
    out = {}
    for k, v in fits_header.items():
        if type(v) is not fits.card.Undefined:
            out[k] = v
        else:
            out[k] = None
    return out

def find_time_range(fits_table_header, injected_row):
    epoch = injected_row["i_epoch"]
    period = injected_row["i_period"]
    dur = injected_row["i_dur"] / 24.
    t_start, t_stop = fits_table_header["TSTART"], fits_table_header["TSTOP"]
    # Find how many periods since epoch is the transit in the light curve.
    num_period = (t_start - epoch) // period + 1  # VERY IMPORTANT OFF BY ONE.
    mid_transit = epoch + period * num_period
    half_width = dur / 2 + BOUND_TOL  # Include an extra hour (2 long cadences)
    start = mid_transit - half_width
    stop = mid_transit + half_width
    logging.info("The start and stop period for {} is {}, {}".format(fits_table_header["OBJECT"], start, stop))
    return start, stop

def strip_rows(fits_table, t_start, t_stop):
    rows = []
    header, data = fits_table.header, fits_table.data
    if header["TSTART"] >= t_start or header["TSTOP"] <= t_stop:  # Assumes that table is sorted by time.
        logging.warning("t_start or t_stop out of light curve bounds")
    for r in data:
        if t_start <= r["TIME"] and r["TIME"] <= t_stop:
            rows.append(r)
    return rows

def strip_cols(fits_rows, metadata):
    data_rows = []
    for r in fits_rows:
        data_rows.append((r["TIME"], r["SAP_FLUX"]))
    t = table.Table(rows=data_rows, names=("TIME", "SAP_FLUX"),
        dtype=("f8", "f8"), meta=metadata)
    return t

def gather_data(filename, injected_table, injected_table_index):
    with fits.open(filename) as hdulist:
        kic_id = injection.parse_injected_filename(filename)
        injected_row = injected_table[injected_table_index[kic_id]]
        start, stop = find_time_range(hdulist[1].header, injected_row)
        rows = strip_rows(hdulist[1], start, stop)
        t = strip_cols(rows, dictify(hdulist[1].header))
    return t

def main():
    injected = injection.parse_injected_table("/mnt/data/meta/kplr_dr25_inj1_plti.txt")
    index = injection.index_injected_table(injected)
    for i in sys.argv[1:]:
        filename = path.abspath(i)
        # "/mnt/data/INJ1/kplr011183555-2011271113734_INJECTED-inj1_llc.fits.gz"
        t = gather_data(filename, injected, index)
        root = path.basename(filename).split(".")[0]
        ascii.write(t, root + "_quicklook.ecsv", format='ecsv')

if __name__ == "__main__":
    main()
