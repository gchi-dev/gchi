# gchi

Calculate the Global Climate Health Index (GCHI): get health-relevant climate hazard indices from gridded data. Give it a dictionary of xarray DataArrays and it returns exceedance fractions and severity levels (0-4) for 27 metrics across heat, cold, fire, air quality, drought, disease, and extreme weather, on a scale that is comparable across hazard types. Outputs are grid cell resolved and aggregated to annual statistics.

Built for CMIP6 outputs, but equally applicable to observational products. Runs chunked/lazy through xarray + dask wherever the math allows it (most of it does; FWI does not, see notes below).  

Note. The GCHI methodology and CMIP6 analysis paper is currently under submission, but will be linked once published. For any questions about methodology in the meantime, please open an issue. 

## Install

```bash
conda create -n gchi -c conda-forge xesmf esmpy
pip install gchi
```

xesmf and dependencies need to be installed with conda

## Quick start
to see input structure run the `gchi.show_expected_ds_format()`  

```python
import gchi

# study period data: dict of DataArrays keyed by CMIP6 shortname
# if your data have difference naming conventions, please map to CMIP6 shortname   
ds_dict = {"tasmax": tasmax_da, "hurs": hurs_da, "pr": pr_da, ...}

results = gchi.calculate_all(ds_dict)
results.summary() # summarizes what ran, what got skipped, what failed

results["AT"] # ex. apparent temperature severity levels
```

Every function also runs standalone if you only want one metric:

```python
gchi.AT(ds_dict)
gchi.FWI(ds_dict)
```

Data gets auto-prepped (chunked, optionally regridded/masked) the first time it's
needed and never re-prepped after that, whether you call `calculate_all()` or an
individual metric directly.

## Base period

Metrics that need a historical baseline (HWF, TNXp, SPI, SMSXp, PR1day, PR5day) need
`base_dict` passed in explicitly - built from your own historical data via
`calculate_base_period_percentiles()`. `gchi` will never guess at a base period from
your study-period data. If you don't pass one, those six metrics get skipped.

`calculate_all()` will run `calculate_base_period_percentiles()` for you automatically if `base_dict` is still raw historical data:

```python
base_dict = {"tas": hist_tas, "tasmin": hist_tasmin, "pr": hist_pr}
results = gchi.calculate_all(ds_dict, base_dict)
```

Calling individual metrics directly (`gchi.TNXp(ds_dict, base_dict)`) skips that detection step, so base_dict needs to already be in percentile form there -> run `calculate_base_period_percentiles()` yourself first:  

```python
base_dict = gchi.calculate_base_period_percentiles(tas=hist_tas, tasmin=hist_tasmin, pr=hist_pr)
gchi.TNXp(ds_dict, base_dict)
```

## Category and composite averages

Every metric shares the same 0-4 severity scale, which means you can average across
metrics and across hazard categories directly:

```python
cat_ds = gchi.category_averages(results) 
composite = gchi.composite_average(cat_ds)
```

Equal weighting by default. Pass `metric_weights` / `category_weights` if you want
some metrics or categories to count more than others: see docstrings for the exact
weighting behavior (it's a relative weighted mean, weights don't need to sum to 1)  

## Reference data

A handful of metrics need external reference files (infrequent-burning mask,
environmental zones for FWI thresholds, aridity mask, land mask, default 1x1 target
grid, MDA8 scale factor). These are hosted on Zenodo and get downloaded and cached
automatically the first time they're needed. These live in `~/.local/share/gchi`. Override with `GCHI_DATA_DIR`.   
You pass your own file path to any of the relevant arguments to skip the default entirely.

## Logging

Quiet by default: only warnings (fallback behavior, guessed units, skipped metrics)
and errors show. Progress messages are opt-in:

```python
gchi.set_verbose(True)
# or pass verbose=True to calculate_all() / prepare_inputs() directly
```

## Things worth knowing

- **FWI does not chunk.** It's a sequential time-step calculation, so it loads data year-by-year internally. Everything else should chunk fine.
- **WBT/WBGT prefer `huss` over `hurs`, but only if the resolution actually matches**
  `tasmax`. If you've got monthly `huss` sitting next to daily everything else (common
  if you pulled `huss` for O3/PM2.5), it'll fall back to deriving humidity from `hurs`
  instead of crashing.
- **`calculate_all` never crashes on one bad metric.** Missing inputs are skipped. Errors are caught, logged, and recorded in `results.failed`, and everything
  else keeps running.


## Citation

If you use this in published work, please cite the relevant release and, if you're describing the software itself, the JOSS paper
(`paper/paper.md` in this repo).

## License

See [LICENSE](LICENSE).
