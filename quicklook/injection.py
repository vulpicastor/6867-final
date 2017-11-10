from astropy.io import ascii
from astropy.io import fits
from os import path

def parse_injected_table(filename):
    injected = ascii.read(filename, format="ipac")
    return injected

def index_injected_table(injected_table):
    injected_index = {}
    for i, row in enumerate(injected_table):
        injected_index[row["KIC_ID"]] = i
    return injected_index

def parse_injected_filename(filename):
    """-> KIC ID of the file"""
    basename = path.basename(filename)
    return int(basename[4:13])
