"""
download and cache default reference data files (masks,
target grid, scale factors) from zenodo.
inspired by the cartopy  natural earth shapefile workflow (https://foundations.projectpythia.org/core/cartopy/cartopy/)

source: GCHI v1 Supporting Datasets, https://zenodo.org/records/19239161
        (Elling, M. - CC BY 4.0)

files are downloaded once into the local cache directory and reused after
that 

never re-downloaded unless the cache is cleared or the checksum
doesn't match. users can always bypass this entirely by passing their own
file path to the relevant function argument (fire_mask_file=, VBD_mask_file=,
model_grid_file=, etc.) 

this module is only consulted when that argument
is left at its "default" 
"""

import os
import hashlib
import urllib.request
from ._log import logger

_ZENODO_BASE_URL = "https://zenodo.org/records/19239161/files"

# checksums copied directly from the zenodo file listing
_DATA_FILES = {
    "model_grid":         ("gchi_targetgrid_1x1_global_v1.nc",     "c27ceb0cdd263b2b0e4c610d9649d522"),
    "land_mask":          ("gchi_land_mask_anytouch_v1.nc",        "c1781406cfb2230ab0bb6799e9250fe1"),
    "environmental_zone": ("gchi_GEnS_envzones_1x1_v1.nc",         "225ae4b50106e81de906a947b7c30ffd"),
    "fire_mask":          ("gchi_infreq_burning_mask_v1.nc",       "80c69c270af4aac5a0d748c1aef8b8e2"),
    "vbd_mask":           ("gchi_aridity_mask_1x1_v1.nc",          "7b922e82119b603766921da77f594974"),
    "mda8_scale":         ("gchi_mda8_daily_scalefac_v1.nc",       "3023d033048b580b3e2540cde22a21ae"),
}


def _get_cache_dir():
    """local cache directory - override with the GCHI_DATA_DIR env var"""
    cache_dir = os.environ.get("GCHI_DATA_DIR", os.path.expanduser("~/.local/share/gchi"))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_default_data_file(name, base_url=None):
    """
    return a local path to the default reference file for `name`, downloading
    and caching it the first time it's needed. subsequent calls reuse the
    cached copy.

    Params
    name : str
        one of the keys in _DATA_FILES (e.g. "fire_mask", "vbd_mask")
    base_url : str, optional
        override the zenodo base url (mainly for testing against a mock host)
    """
    if name not in _DATA_FILES:
        raise ValueError(f"no default data file registered for '{name}' - expected one of {list(_DATA_FILES)}")

    filename, checksum = _DATA_FILES[name]
    cache_dir = _get_cache_dir()
    local_path = os.path.join(cache_dir, filename)

    if os.path.exists(local_path):
        if checksum is None or _md5(local_path) == checksum:
            return local_path
        logger.warning(f"{filename}: cached file failed checksum verification - re-downloading...")

    url = f"{base_url or _ZENODO_BASE_URL}/{filename}?download=1"
    logger.info(f"downloading default {name} file (first time only)...")
    logger.info(f"    {url}")
    logger.info(f"    -> {local_path}")

    tmp_path = local_path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp_path)
        if checksum is not None and _md5(tmp_path) != checksum:
            os.remove(tmp_path)
            raise RuntimeError(f"downloaded {filename} failed checksum verification")
        os.replace(tmp_path, local_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return local_path


def clear_data_cache():
    """delete all cached default data files, forcing a fresh download next time"""
    cache_dir = _get_cache_dir()
    for name, (filename, _) in _DATA_FILES.items():
        path = os.path.join(cache_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"removed {path}")
