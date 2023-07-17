import os
import datetime as dt
from urllib.request import urlopen

from ..utils import downloadfile

def download_gcmt_catalog(catalog_filename="gcmt.ndk"):
    # Get catalog from 1976 to 2017
    url_cat = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec17.ndk"


    # Download the catalog
    print(f"Downloading {url_cat}")
    downloadfile(url_cat, catalog_filename)

    # Get monthly catalog from 2018 on
    ext = '.ndk'
    link = 'https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/'

    thisyear = dt.datetime.now().year
    thismonth = dt.datetime.now().month

    with open(catalog_filename, "a") as catalogfile:

        for year in range(2018, dt.datetime.now().year + 1):

            yy = f"{year}"[-2:]

            for month in ["jan", "feb", "mar", "apr", "may", "jun",
                          "jul", "aug", "sep", "oct", "nov", "dec"]:

                if (year == thisyear) \
                        and (month == thismonth):
                    break
                else:

                    url_monthly = f"{link}{year}/{month}{yy}{ext}"
                    print(f"Downloading {url_monthly}")

                    try:
                        catalogfile.write(
                            urlopen(url_monthly).read().decode('utf-8'))
                    except Exception as e:
                        print("    ... failed: ", e)