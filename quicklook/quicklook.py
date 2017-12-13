#!/usr/bin/python3

from astropy import table
from astropy.io import fits, ascii
import logging
import numpy as np
from os import path
import random
import sys
import traceback

import injection

# Whether to randomize padding. if True, will be randomly selected between
# BOUND_TOL_LOWER and BOUND_TOL_UPPER. If False, will use fixed BOUND_TOL.
# True for LSTM, False for SVM.
RANDOMIZE_BOUND = True
BOUND_TOL = 6. / 24  # day; this is equivalent to 12 long cadences.
BOUND_TOL_UPPER = 1.  # day
BOUND_TOL_LOWER = 3./24  # day
NEG_UPPER = 2.  # day; upper bound for time span of negative samples.
NEG_LOWER = 8. / 24  # day; lower bound for time span of negative samples.

def dictify(fits_header):
    """-> dictionary representation of the FITS header.

    This is necessary because undefined cards can't be serialized and need to
    be turned in to Nones.
    """
    out = {}
    for k, v in fits_header.items():
        if type(v) is not fits.card.Undefined:
            out[k] = v
        else:
            out[k] = None
    return out

def find_time_range(table_start, table_stop, epoch, period, dur, name="Unknown"):
    # Find how many periods since epoch is the transit in the light curve.
    num_period = (table_start - epoch) // period + 1  # VERY IMPORTANT OFF BY ONE.
    # Calculate the start and stop times of the transit.
    mid_transit = epoch + period * num_period  # Time of mid-transit
    half_width = dur / 2.
    transit_start = mid_transit - half_width
    transit_stop = mid_transit + half_width
    logging.info("The table start and stop period for %s is %s, %s", name, table_start, table_stop)
    logging.info("The transit start and stop period for %s is %s, %s", name, transit_start, transit_stop)
    if table_start >= transit_start or transit_stop >= table_stop:
        logging.error("Transit outside light curve range for %s!", name)
    # Sanity checking.
    if RANDOMIZE_BOUND:
        # For LSTM, padding should be randomized in order to prevent overfitting based on
        # when the transit starts in the light curve.
        start = transit_start - random.uniform(BOUND_TOL_LOWER, BOUND_TOL_UPPER)
        stop = transit_stop + random.uniform(BOUND_TOL_LOWER, BOUND_TOL_UPPER)
    else:
        # Constant padding otherwise.
        start = transit_start - BOUND_TOL
        stop = transit_stop - BOUND_TOL
    # Check that transit is not clipped by data boundary
    if table_start >= start or stop >= table_stop:
        logging.error("Transit too close to light curve broundary for %s!", name)
        return None, None, None, None
    logging.info("The start and stop period for {} is {}, {}".format(name, start, stop))
    return start, stop, transit_start, transit_stop

def strip_rows(time_col, time_start, time_stop, name="Unknown"):
    # Sanity check: no time value is NaN or inf
    if not np.all(np.isfinite(time_col)):
        logging.error("Non-finite time detected; abandon hope for %s", name)
        return None, None
    # Sanity check: is the table sorted by time?
    for a, b in zip(time_col[:-1], time_col[1:]):
        if a >= b:
            logging.error("Time flew backwards or stood still in %s", name)
            return None, None
    index_start = np.searchsorted(time_col, time_start)
    index_stop = np.searchsorted(time_col, time_stop, side="right")
    return index_start, index_stop

def strip_rows_negative(time_col, time_start, time_stop, name="Unknown"):
    # Choose a duration for the negative sample.
    dur = random.uniform(NEG_LOWER, NEG_UPPER)
    # Randomly find a start and stop time that does not overlap with interval.
    for i in range(100):
        neg_start = random.choice(time_col)
        neg_stop = neg_start + dur
        if not ((neg_start <= time_stop) and (time_start <= neg_stop)):
            break
    else:
        logging.error("Cannot find a negative sample in %s", name)
        return None, None
    return strip_rows(time_col, neg_start, neg_stop, name)

def make_label_column(length, start, stop, col_name):
    out = np.zeros(length)
    out[start:stop] = 1
    return table.Column(data=out, name=col_name)

def strip_cols(fits_table, transit_start, transit_stop, is_eb, dur):
    metadata = dictify(fits_table.header)
    data_rows = []
    start = mid_transit - dur / 2
    stop = mid_transit + dur / 2
    metadata["EB_injection"] = is_eb
    for r in fits_rows:
        # in_transit is 0 if outside transit and 1 if inside transit.
        in_transit = int((start <= r["TIME"]) and (r["TIME"] <= stop))
        eb_injection = int(is_eb and in_transit)
        data_rows.append((r["TIME"], r["SAP_FLUX"], in_transit, eb_injection))
    t = table.Table(rows=data_rows, names=("TIME", "SAP_FLUX", "IN_TRANSIT", "EB_injection"),
        dtype=("f8", "f8", "i4", "i4"), meta=metadata)
    return t

def gather_data(filename, injected_table, injected_table_index):
    with fits.open(filename) as hdulist:
        # Extract KIC ID part of the file name.
        kic_id = injection.parse_injected_filename(filename)
        injected_i = injected_table_index[kic_id]
        # Extract row metadata
        epoch = injected_table["i_epoch"][injected_i]
        period = injected_table["i_period"][injected_i]
        dur = injected_table["i_dur"][injected_i] / 24.  # hour
        table_start = hdulist[1].header["TSTART"]
        table_stop = hdulist[1].header["TSTOP"]
        name = hdulist[1].header["OBJECT"]
        # Find when the transit starts and stops in the FITS file.
        start, stop, transit_start, transit_stop = find_time_range(table_start, table_stop, epoch, period, dur, name)
        # When find_time_range fails, you know you are fucked.
        if start is None:
            return None, None
        time_col = hdulist[1].data["TIME"]
        # Strip out the rows that are actually transts.
        i_start, i_stop = strip_rows(time_col, start, stop, name)
        if i_start is None:
            return None, None
        # Also generate the negative samples
        i_neg_start, i_neg_stop = strip_rows_negative(time_col, start, stop, name)
        # Find transit start and stop indices
        i_transit_start, i_transit_stop = strip_rows(time_col, transit_start, transit_stop, name)
        # Extract information of whether an eclipsing binary is being simulated
        is_eb = injected_table["EB_injection"][injected_i]
        if is_eb:
            transit_col = make_label_column(len(time), i_transit_start, i_transit_stop, "IN_TRANSIT")
            eb_col = table.Column(data=np.zeros(len(time)), name="EB_injection")
        else:
            transit_col = table.Column(data=np.zeros(len(time)), name="IN_TRANSIT")
            eb_col = make_label_column(len(time), i_transit_start, i_transit_stop, "EB_injection")
        # Only preserve the TIME and SAP_FLUX columns of the light curve.
        # Also add tag for whether the mid transit point has passed.
        t = strip_cols(rows, dictify(hdulist[1].header), mid_transit, is_eb, dur)
        if i_neg_start is None or is_eb:
            neg_t = None
        else:
            neg_t = strip_cols(neg_rows, dictify(hdulist[1].header), mid_transit, 0, dur)
    return t, neg_t

def main():
    logging.warning("Making sure that warning shots are fired")
    # Load the injected transits data table.
    injected = injection.parse_injected_table("/mnt/data/meta/kplr_dr25_inj1_plti.txt")
    index = injection.index_injected_table(injected)
    # Iterate through all filenames given as command line arguments.
    for i in sys.argv[1:]:
        logging.info("Processing %s", i)
        try:
            filename = path.abspath(i)
            # Example of an okay light curve:
            # "/mnt/data/INJ1/kplr011183555-2011271113734_INJECTED-inj1_llc.fits.gz"
            root = path.basename(filename).split(".")[0]
            t, neg_t = gather_data(filename, injected, index)
            if t is not None:
                ascii.write(t, root + "_quicklook.ecsv", format='ecsv')
            if neg_t is not None:
                ascii.write(neg_t, root + "_quicklook_negative.ecsv", format='ecsv')
        except Exception as e:
            logging.error("Error while processing file %s: %s", i, e)
            logging.error("Traceback: %s", traceback.format_exc())

if __name__ == "__main__":
    logging.basicConfig(filename="output.log", level=logging.INFO)
    main()
