# Source-specific permissions and attribution

Own research code: MIT (`LICENSE`). Own model outputs, original figures and
documentation: Creative Commons Attribution 4.0 International
(https://creativecommons.org/licenses/by/4.0/). Neither grant replaces upstream
rights or grants permission to reuse third-party logos or trademarks.

| Supplied content | Original source and terms | Extraction/transformation |
| --- | --- | --- |
| Cluster-scale location and hydrogen estimates | European Hydrogen Observatory, Clean Hydrogen Joint Undertaking. [Official workbook](https://observatory.clean-hydrogen.europa.eu/sites/default/files/2025-09/Hydrogen%20production%20capacity%202024%20%28Update%202025%29_0_0.xlsx); [legal notice](https://observatory.clean-hydrogen.europa.eu/index.php/legal-notice): reproduction authorised provided the source is acknowledged, unless stated otherwise. | 17 ammonia-linked, non-electrolysis records combined into 15 city clusters. Estimated output weights coordinates and scales equivalent service. The original workbook and logos are not redistributed. |
| Hourly PV capacity factor | European Commission Joint Research Centre, PVGIS 5.3, ERA5. [API](https://re.jrc.ec.europa.eu/api/v5_3/seriescalc); [usage conditions](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/general-information/usage-conditions-data-protection_en): free information without restrictions on use. | Modeled power for 1 kWp divided by 1000 W. PVGIS hourly midpoints aligned to UTC interval starts. Original response hashes retained. |
| Hourly wind speed at 100 m | [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api), ERA5/Copernicus ECMWF; [data licence](https://open-meteo.com/en/licence): CC BY 4.0. [New API calls](https://open-meteo.com/en/terms) are subject to applicable service terms and call limits. | Exact numerical field extracted and UTC-aligned, with original response hashes. Subsequent generic 3/12/25 m/s wind-power curve is a research assumption, not observed turbine output. |
| Model parameters extracted from reports | Hystories D7.1/D7.2, IRENA and cited engineering literature. Official URLs are recorded in `data_manifest/sources.tsv`. | Only necessary attributed numerical parameters and references are supplied. Complete reports, publisher PDFs and copyrighted figures are not redistributed or relicensed. |
| Vendored plotting presets | [SciencePlots](https://github.com/garrettj403/SciencePlots), commit b9b16959570bd2fbc9ff5118bacc423c3bddd592, MIT. | Three unmodified style files and original license in `src/vendor/scienceplots/`. Research-specific sizing, accessibility and layout overrides are in the figure script. |

Retrieval/extraction dates and request parameters are in the supplied manifests.
The hourly caches cover 2005–2023; they are reanalysis/model data, not measured
factory operation. No personal data, private industrial data, confidential
network/port/cavern data or operator credentials are included.

Licensing pages were checked on 2026-10-01. Cite the respective original source
when reusing its extracted fields, acknowledge the transformations above, and
retain this attribution file when redistributing the caches.
