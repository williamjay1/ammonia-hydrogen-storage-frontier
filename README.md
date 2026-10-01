# Ammonia process flexibility and hydrogen-storage service frontiers

Reproducible engineering model, public-data extracts, complete numerical results,
and publication figures accompanying **Process flexibility and product buffering
expand the deliverability-constrained service capacity of hydrogen storage for
renewable ammonia**, by Siyu Wang and Junjie Zhang. Manuscript under preparation;
no publication or acceptance is implied.

## Scientific scope

The sample contains 15 public city-scale clusters in France, Italy, the
Netherlands, Poland and Spain, formed from 17 ammonia-linked hydrogen-production
records. These are not a census of EU-27 factories. Hourly PVGIS/Open-Meteo ERA5
inputs cover 2005–2023, with sizing on 2006–2014 and separate annual evaluation
replays for 2015–2023. Weather and PV output are reanalysis/model products;
industrial demand, process envelopes and reference storage units are scenarios.

At equal normalized service and equal non-storage sizing rules, the primary
60%-of-nameplate synthesis floor with a 24-hour product buffer gives a median
**21.1% continuous service-capacity gain** over three weather/scale anchors and
three years, and **25.3% minimum withdrawal-rating relief** at a common working-gas
volume. Across all 135 location–year pairs the median unconstrained hydrogen
inventory reduction is **17.0%**. These quantities have different denominators
and must not be interchanged. No actual plant reliability, cavern availability,
construction savings, costs, LCA or commercial production gains are estimated.
Rigid, flexible, adverse and infeasible sensitivity cases are all retained.

## Reproduce

Python 3.12 was used. Install the exact recorded dependencies in a virtual
environment using `pip install -r requirements.txt`.

```bash
python src/reproduce_current_article.py --mode verify
python -m pytest tests -q
python src/reproduce_current_article.py --mode primary --work-root ./work/annual-check --site-ids spain_huelva --years 2019
python src/build_nature_figures.py --results-root ./results --output-dir ./work/figures
```

The annual example freshly solves three full-hourly process cases. Omit the
location/year restrictions to run all 405 primary cases. `docs/reproduction.md`
describes frontier, fixed-volume, threshold and additional sensitivity reruns.
On Windows, choose a writable D-drive `--work-root`: the entry point deliberately
rejects C-drive computation. On Linux/macOS, an explicit writable relative or
absolute work directory is supported. Inputs in this repository are not modified.

## Contents

| Directory | What is supplied |
| --- | --- |
| `datasets/` | Public-derived cluster table and 15 exact hourly numerical caches |
| `data_manifest/` | Official sources, retrieval parameters and extraction checks |
| `src/`, `tests/` | Frozen model, portable rerun entry point and numerical tests |
| `results/primary/` | 405 primary cases; matched results for both process floors |
| `results/frontier/`, `results/common_volume/` | 27 continuous and 27 fixed-volume cases |
| `results/threshold/`, `results/threshold_provenance/` | All 90 classifications and staged calculation provenance |
| `results/review_added_diagnostics/` | 90 additional wind/volume/grid cases, including six failures |
| `figures/` | Vector PDF/SVG/EPS and native 1200-dpi PNGs with layout checks |
| `tables/` | The six publication tables as CSV and isolated native LaTeX table bodies |
| `docs/` | Reproduction, variable definitions, licensing and publication checks |

`manifest_sha256.json` verifies the released scientific files. Historical local
paths in JSON provenance are replaced with explicit archival-path labels, while
all original source hashes, retrieval dates, numerical values and mathematical
source files are retained. The original third-party workbook, reports, papers,
author emails, manuscript files and internal editorial records are not bundled.

## Attribution and licensing

Own research code is under the MIT license. Own numerical outputs and figures
are CC BY 4.0. Upstream source data retain their original terms; **the code license
does not relicense third-party data**. See `DATA_LICENSES.md` for exact source
links, permissions and transformations. Weather data acknowledge the European
Commission Joint Research Centre/PVGIS, Open-Meteo and Copernicus/ECMWF ERA5;
location/scale extracts acknowledge the European Hydrogen Observatory/Clean
Hydrogen Joint Undertaking.

Figure styling uses the community [SciencePlots](https://github.com/garrettj403/SciencePlots)
`science`, `nature` and `no-latex` presets pinned to commit
`b9b16959570bd2fbc9ff5118bacc423c3bddd592`, with its MIT license preserved.
This is not an official Nature plotting library or an endorsement by Nature.
Vector artwork is resolution-independent; the PNG export is rendered natively
at 1200 dpi, not enlarged from a low-resolution image.

## Cite and archive

Use `CITATION.cff` to cite this software/data release. The repository and tagged
release provide a public versioned record. No archival DOI has yet been minted;
do not cite a fictitious DOI. A later external archival deposit can add a DOI
without changing the reported scientific results.
