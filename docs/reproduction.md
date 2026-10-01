# Reproducing the current ammonia storage-capability study

This supplement contains the current model and results, not the retired
country-reference/template studies. Public weather is modeled/reanalysis data.
Hourly product service, process capability and the storage archetype are stated
scenarios, not observed operating records.

## Supplied materials

- `datasets/eho_ammonia_site_clusters_v1_20260930.csv`: 17 ammonia-linked
  conventional-H2 records aggregated into 15 city clusters. Estimated annual
  H2 output is a scale parameter, not measured ammonia output or orders.
- `datasets/current_article_weather_cache`: exact numerical PV-factor and
  100-m wind-speed fields extracted from public PVGIS 5.3/Open-Meteo ERA5
  responses, 2005–2023. UTC interval-start alignment is explicit. The original
  response hashes and extraction checks are recorded. No turbine yield is
  represented by the generic wind curve.
- `results/primary`: all 405 location–year–process cases and 135 pairs per floor.
- `results/frontier`: all 27 continuous storage-capability cases and scale
  diagnostics. `results/common_volume`: all 27 fixed-volume rate comparisons.
- `results/threshold`: all 90 one-unit process-floor sensitivity classifications,
  including failures. Complete staged numerical provenance is retained.
- `results/review_added_diagnostics`: all 90 additional configurations: 36
  wind-mix frontier, 36 medium/large common-volume and 18 grid-boundary cases.
  Six zero-grid failures are retained. The plan was recorded before execution.
- `src` and `tests`: mathematical model, acquisition/transformation functions,
  numerical formulations, independent bounds and relevant tests.
- Source/access/licence table, environment versions and relative-path hashes.

The original EHO workbook and complete Hystories/IRENA reports are not
redistributed. Their official download links and model extractions are supplied.
PV values acknowledge the European Commission Joint Research Centre/PVGIS;
wind data acknowledge Open-Meteo and are supplied under its CC BY 4.0 terms.
The free API's non-commercial call conditions apply to new retrieval, not a
claim that all source material has a single blanket licence. No personal data
or proprietary industrial/network data are included.

## Fast integrity check

Use Python 3.12 with the supplied package versions (NumPy, pandas, SciPy,
matplotlib, requests, openpyxl and time-zone data). From the extracted package:

```
python src/reproduce_current_article.py --mode verify
```

The relative-path manifest checks every supplied file; the weather check reads
all 15 hourly caches. The original run manifests retain original paths and raw
hashes for provenance; those historical paths are not required by this check.

## Re-run a complete annual benchmark

Choose a new explicit work directory. On the authors' Windows setup, use D:
for computation; never point the work directory at the immutable raw repository.

```
python src/reproduce_current_article.py --mode primary --work-root D:/MLWork/project12_reproduction --site-ids spain_huelva --years 2019
```

This produces all three full-hourly process cases for the requested annual
benchmark, using the same original mathematical code and parameters. Omitting
site/year arguments runs all 405 cases and can require hours. This is a fresh
run, not a resumed original manifest. Only filesystem roles are redirected in
memory. The two cached weather fields are rehydrated into the minimal original
reader schema; this changes response-container hashes, not model data. New
manifests record the actual reconstructed inputs.

```
python src/reproduce_current_article.py --mode frontier --work-root D:/MLWork/project12_frontier_reproduction
python src/reproduce_current_article.py --mode common-volume --work-root D:/MLWork/project12_rate_reproduction
python src/reproduce_current_article.py --mode threshold --work-root D:/MLWork/project12_threshold_reproduction
python src/reproduce_current_article.py --mode figures --work-root D:/MLWork/project12_figure_reproduction
python src/reproduce_current_article.py --mode review-wind --work-root D:/MLWork/project12_review_wind
python src/reproduce_current_article.py --mode review-volume --work-root D:/MLWork/project12_review_volume
python src/reproduce_current_article.py --mode review-grid --work-root D:/MLWork/project12_review_grid
```

The frontier and common-volume commands each re-run 27 declared cases. The
threshold command re-runs the 90-case grid with the original equivalent
reduced-form, full-matrix-checked LP; an optional `--anchor low`, `median` or
`high` restricts this to 30 cases. This numerical backend can be slow. Original
mixed-backend execution records, including the expanded-form feasibility
witnesses, are retained with the supplied results; the fresh calculation
records its own backend and does not resume or modify those outputs. The
figures command uses the complete supplied outputs. Scenario definitions and
configuration manifests specify the exact ten-factor/three-anchor grid.

The review-added modes repeat the predeclared one-factor blocks using exactly
rehydrated weather and unchanged mathematical source. A quick full-year pair
can use `--max-new-cases 2`; rerunning the same command without that limit
resumes only when source/input hashes match. These 90 cases are distinct from
the original 90 threshold cases; they are not new observed installations.

## Mathematical and inferential boundaries

Normalized hourly service is 1 tonne NH3-equivalent. Renewable, electrolyser,
grid and synthesis nameplates scale linearly with service. A continuous
one-unit service-capacity gain is therefore a gain within that standardized
design family, not increased output at a fixed existing plant. Cases compared
at the same service have identical non-storage nameplates.

The two stock balances, electricity boundary and cyclic ramp edge must be
preserved. The primary 60% nameplate floor equals 0.9 normalized tonnes/h;
40% equals 0.6. A 24-hour product-buffer ceiling changes jointly with process
flexibility relative to the rigid zero-buffer case. WTIR=2 links injection to
withdrawal. Perfect foresight and annually optimized cyclic initial stocks
are favorable scheduling assumptions, not reliability validation. Replays
are separate calendar years, not one continuous multiyear trajectory.

Free-inventory relief is bounded by 24 service hours under the checked grid
reconstruction condition. The interval cuts are necessary, not sufficient.
Reference-module counts are conceptual adequacy outcomes: no geology,
pressure-dependent flow, cushion gas, outage rates, linepack, costs, LCA or
investment savings are inferred. All adverse sensitivity cases remain in the
supplied tables. Intermediate constructive schedules are marked as numerical
feasibility witnesses, not observations or a separate optimization algorithm.
