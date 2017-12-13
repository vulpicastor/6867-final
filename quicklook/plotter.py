from astropy.io import ascii
import matplotlib.pyplot as plt
from os import path
import sys

def plot_table(ascii_table, filename):
    time = ascii_table["TIME"]
    flux = ascii_table["SAP_FLUX"]
    transit = ascii_table["IN_TRANSIT"]
    title = ascii_table.meta["OBJECT"]
    transit = transit * 100 - 50
    plt.scatter(time, flux, c=transit, cmap=plt.get_cmap("bwr"))
    plt.title(title + " SIMULATED")
    plt.savefig(filename)
    plt.show()

def main():
    filename = sys.argv[1]
    lc_table = ascii.read(filename)
    root = path.basename(filename).split(".")[0]
    plot_table(lc_table, root + ".pdf")

if __name__ == "__main__":
    main()
