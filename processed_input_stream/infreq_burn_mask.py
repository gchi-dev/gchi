import numpy as np
import xarray as xr
import netCDF4 as nc
#from tqdm import tqdm

# ============================================================================================
# 0. OPTIONS
# ============================================================================================
# INPUTS
input_grid_file = "/projects/mael9842/fwi_mask_data/gchi_targetgrid_1x1_global_20260326.nc"
file_esa_cci_lc = "/projects/mael9842/fwi_mask_data/C3S-LC-L4-LCCS-Map-300m-P1Y-2016-v2.1.1.nc"

# FLAGS of ESA-CCI-LC that will be used for categories
list_infreq_burning = [
    "bare_areas",
    "bare_areas_consolidated",
    "bare_areas_unconsolidated",
    "water",
    "snow_and_ice",
    "sparse_vegetation",
    "sparse_tree",
    "sparse_shrub",
    "sparse_herbaceous",
]
list_water = ["water"]

# OUTPUT
file_save = "/projects/mael9842/fwi_mask_data/fwi_mask_1x1_global.nc"

# ============================================================================================
# 1. LOAD INPUT GRID
# ============================================================================================
print("=" * 80)
print("STEP 1: Loading input grid...")
print("=" * 80)
grid_data = xr.open_dataset(input_grid_file)

# Extract lat/lon - adjust these variable names to match your file
lat = grid_data['lat'].values
lon = grid_data['lon'].values
print(f"  Grid shape: {len(lat)} lat x {len(lon)} lon")

# Create bounds if they don't exist
if 'lat_bnds' not in grid_data:
    print("  Creating lat_bnds...")
    lat_bnds = np.array([
        [l - 0.5 * (lat[1] - lat[0]), l + 0.5 * (lat[1] - lat[0])]
        for l in lat
    ])
else:
    lat_bnds = grid_data['lat_bnds'].values
    print("  Using existing lat_bnds")

if 'lon_bnds' not in grid_data:
    print("  Creating lon_bnds...")
    lon_bnds = np.array([
        [l - 0.5 * (lon[1] - lon[0]), l + 0.5 * (lon[1] - lon[0])]
        for l in lon
    ])
else:
    lon_bnds = grid_data['lon_bnds'].values
    print("  Using existing lon_bnds")

print("✓ Grid loaded\n")

# ============================================================================================
# 2. PREPARING ESA-CCI-LC_Land-Cover-Maps
# ============================================================================================
print("=" * 80)
print("STEP 2: Loading ESA-CCI land cover data...")
print("=" * 80)

ds_cci = nc.Dataset(file_esa_cci_lc)
lc_var = ds_cci.variables["lccs_class"]

# get lat/lon from file
lat_cci = ds_cci.variables["lat"][:]
lon_cci = ds_cci.variables["lon"][:]
print(f"  Shape: lat={len(lat_cci)}, lon={len(lon_cci)}")

# checks
if ds_cci.variables["lat"].units not in ["degrees_north"]:
    raise Exception("Incorrect units for grid")
if ds_cci.variables["lon"].units not in ["degrees_east"]:
    raise Exception("Incorrect units for grid")

# preparing flags
flag_meanings = lc_var.flag_meanings
flag_values   = lc_var.flag_values
flag_names = str.split(flag_meanings, " ")
dico_names_values = {
    flag_names[i]: val for i, val in enumerate(flag_values)
}
print(f"  Found {len(flag_names)} land cover classes")

infreq_values = np.array([dico_names_values[n] for n in list_infreq_burning])
water_values  = np.array([dico_names_values[n] for n in list_water])

# cell areas: 1D, no 2D array needed
cos_lat_cci = np.cos(np.deg2rad(lat_cci))
cell_area_1d = cos_lat_cci * (510072000 * 1.0e6) / (cos_lat_cci.sum() * len(lon_cci))  # m2

print("✓ Land cover processing complete\n")

# ============================================================================================
# 3. AGGREGATING
# ============================================================================================
print("=" * 80)
print("STEP 3: Aggregating to target grid...")
print("=" * 80)

# matching for aggregation
print("  Creating lat matching indices...")
match_lat = np.array([
    np.where(
        (lat_bnds[:, 0] <= lat_cci_i) & (lat_cci_i <= lat_bnds[:, 1])
    )[0][0]
    for lat_cci_i in lat_cci
])

print("  Creating lon matching indices...")
match_lon = np.array([
    np.where(
        (lon_bnds[:, 0] <= lon_cci_j) & (lon_cci_j <= lon_bnds[:, 1])
        | (lon_bnds[:, 0] <= lon_cci_j + 360) & (lon_cci_j + 360 <= lon_bnds[:, 1])
    )[0][0]
    for lon_cci_j in lon_cci
])

# aggregating in vectorized lat chunks — never materializes the full 2D array
print("  Aggregating infrequent burning surface...")
print("  Aggregating water surface...")
print("  Aggregating total surface...")
out_infreq = np.zeros(len(lat) * len(lon))
out_sea    = np.zeros(len(lat) * len(lon))
out_tot    = np.zeros(len(lat) * len(lon))

has_time = (lc_var.ndim == 3)
batch = 500

for i in range(0, len(lat_cci), batch):
    lc_chunk   = np.array(lc_var[0, i:i+batch, :] if has_time else lc_var[i:i+batch, :])  # (batch, nlon)
    ml_chunk   = match_lat[i:i+batch]                                                        # (batch,)
    area_chunk = cell_area_1d[i:i+batch]                                                     # (batch,)

    flat_idx  = (ml_chunk[:, None] * len(lon) + match_lon[None, :]).ravel()                 # (batch*nlon,)
    flat_area = np.repeat(area_chunk, len(lon_cci))                                          # (batch*nlon,)

    out_tot    += np.bincount(flat_idx, weights=flat_area,                                                        minlength=len(lat)*len(lon))
    out_infreq += np.bincount(flat_idx, weights=flat_area * np.isin(lc_chunk.ravel(), infreq_values),            minlength=len(lat)*len(lon))
    out_sea    += np.bincount(flat_idx, weights=flat_area * np.isin(lc_chunk.ravel(), water_values),             minlength=len(lat)*len(lon))
    del lc_chunk, flat_idx, flat_area

ds_cci.close()

out_tot    = out_tot.reshape(len(lat), len(lon))
out_infreq = out_infreq.reshape(len(lat), len(lon))
out_sea    = out_sea.reshape(len(lat), len(lon))

print("✓ Aggregation complete\n")

# ============================================================================================
# 4. CREATE OUTPUT DATASET
# ============================================================================================
print("=" * 80)
print("STEP 4: Creating output dataset...")
print("=" * 80)

DATA_MASK = xr.Dataset()
DATA_MASK.coords["bnds"] = [0, 1]
DATA_MASK.coords["lat"] = lat
DATA_MASK.coords["lon"] = lon

# Add bounds
DATA_MASK["lat_bnds"] = xr.DataArray(
    lat_bnds, coords={"lat": lat, "bnds": [0, 1]}, dims=("lat", "bnds")
)
DATA_MASK["lon_bnds"] = xr.DataArray(
    lon_bnds, coords={"lon": lon, "bnds": [0, 1]}, dims=("lon", "bnds")
)

# archiving - exactly as Quilcaille does
DATA_MASK["area_total"] = xr.DataArray(out_tot,           coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
DATA_MASK["area_land"]  = xr.DataArray(out_tot - out_sea, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
DATA_MASK["area_infreq_burning"]     = xr.DataArray(out_infreq,           coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
DATA_MASK["fraction_infreq_burning"] = xr.DataArray(out_infreq / out_tot, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
DATA_MASK["fraction_water"]          = xr.DataArray(out_sea / out_tot,    coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

# general attributes
DATA_MASK.attrs["info"] = "Spatial information relative to regridded FWI."
DATA_MASK.attrs["source_dataset"] = "ESA-CCI-LC_Land-Cover-Maps"
DATA_MASK.attrs["source_file"] = "C3S-LC-L4-LCCS-Map-300m-P1Y-2016-v2.1.1.nc"

# attributes on variables
DATA_MASK["area_total"].attrs["unit"] = "m2"
DATA_MASK["area_total"].attrs["description"] = "Total surface of the grid cell"

DATA_MASK["area_land"].attrs["unit"] = "m2"
DATA_MASK["area_land"].attrs[
    "description"
] = "Land surface of the grid cell. Only water is excluded here, not water and ice."

DATA_MASK["area_infreq_burning"].attrs["unit"] = "m2"
DATA_MASK["area_infreq_burning"].attrs[
    "description"
] = "Surface of the grid cell considered as infrequent burning."
DATA_MASK["area_infreq_burning"].attrs[
    "list_flags"
] = "Flags considered for infrequent burning: " + ", ".join(list_infreq_burning)

DATA_MASK["fraction_infreq_burning"].attrs["unit"] = "1"
DATA_MASK["fraction_infreq_burning"].attrs[
    "description"
] = "Areal fraction of the grid cell considered as infrequent burning."
DATA_MASK["fraction_infreq_burning"].attrs[
    "list_flags"
] = "Flags considered for infrequent burning: " + ", ".join(list_infreq_burning)

DATA_MASK["fraction_water"].attrs["unit"] = "1"
DATA_MASK["fraction_water"].attrs[
    "description"
] = "Areal fraction of the grid cell covered with water. Snow and ice are excluded."

DATA_MASK["mask_infreq_burning"] = xr.DataArray(
    (out_infreq / out_tot) > 0.80,
    coords={"lat": lat, "lon": lon},
    dims=("lat", "lon")
)
DATA_MASK["mask_infreq_burning"].attrs["unit"] = "1"
DATA_MASK["mask_infreq_burning"].attrs["description"] = (
    "Boolean mask: True where more than 80% of the grid cell is flagged as infrequent "
    "burning (bare areas, water, snow/ice, sparse vegetation). These cells should be "
    "masked out of FWI results."
)
DATA_MASK["mask_infreq_burning"].attrs["threshold"] = "0.80"
DATA_MASK["mask_infreq_burning"].attrs["list_flags"] = (
    "Flags considered for infrequent burning: " + ", ".join(list_infreq_burning)
)
print("✓ Output dataset created\n")

# ============================================================================================
# 5. SAVE
# ============================================================================================
print("=" * 80)
print("STEP 5: Saving output...")
print("=" * 80)
print(f"  Output file: {file_save}")

DATA_MASK.to_netcdf(
    file_save,
    encoding={var: {"zlib": True} for var in DATA_MASK.variables},
)

print("✓ Done!\n")
print("=" * 80)
