"""Build an auditable public ammonia-site sample and acquire hourly weather.

The EHO workbook and API responses are treated as immutable raw inputs on F:.
The site table, manifests, and all later transformations are written on D:.
This is a coverage-limited scenario sample, not an EU ammonia census or plant
operations dataset.
"""

from __future__ import annotations

import argparse
import http.client
import hashlib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd


PROJECT = Path(r"D:\MLWork\project12_h2flex_eu")
SOURCE_XLSX = Path(
    "F:/AcademicData/project12_h2flex_eu/raw/eho_hydrogen_capacity_20260930/"
    "Hydrogen_production_capacity_2024_update_2025.xlsx"
)
RAW_ROOT = Path(
    r"F:\AcademicData\project12_h2flex_eu\raw\eho_ammonia_site_weather_20260930"
)
DERIVED_ROOT = PROJECT / "datasets"
MANIFEST_ROOT = PROJECT / "data_manifest" / "eho_ammonia_sites_2024_v1"
SAMPLE_PATH = DERIVED_ROOT / "eho_ammonia_site_clusters_v1_20260930.csv"

EHO_PAGE = (
    "https://observatory.clean-hydrogen.europa.eu/hydrogen-landscape/"
    "production-trade-and-cost/hydrogen-production"
)
EHO_XLSX_URL = (
    "https://observatory.clean-hydrogen.europa.eu/sites/default/files/2025-09/"
    "Hydrogen%20production%20capacity%202024%20%28Update%202025%29_0_0.xlsx"
)
EHO_LEGAL_URL = "https://observatory.clean-hydrogen.europa.eu/index.php/legal-notice"
PVGIS_DOC_URL = (
    "https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-"
    "information-system-pvgis/using-pvgis-5/api-non-interactive-service_en"
)
PVGIS_HOURLY_DOC_URL = (
    "https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-"
    "information-system-pvgis/using-pvgis-5/pvgis-5-tools/hourly-radiation_en"
)
OPEN_METEO_DOC_URL = "https://open-meteo.com/en/docs/historical-weather-api"
PVGIS_API = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
OPEN_METEO_API = "https://archive-api.open-meteo.com/v1/archive"

H2_PER_T_NH3_KG = 177.55300484072126
EXPECTED_HOURS = 166_536  # 2005–2023, including four leap years.
TIMEZONES = {
    "France": "Europe/Paris",
    "Italy": "Europe/Rome",
    "Netherlands": "Europe/Amsterdam",
    "Poland": "Europe/Warsaw",
    "Spain": "Europe/Madrid",
}
EU_SAMPLE_COUNTRIES = set(TIMEZONES)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_derived_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace a differing derived artifact: {path}")
        return
    with NamedTemporaryFile(dir=path.parent, prefix=".build-", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(content)
        stream.flush()
    try:
        if path.exists():
            raise FileExistsError(f"Refusing to replace an artifact created concurrently: {path}")
        temp.rename(path)
    finally:
        if temp.exists():
            temp.unlink()


def slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_value.lower()).strip("_")


def build_site_sample() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not SOURCE_XLSX.is_file():
        raise FileNotFoundError(f"Public EHO workbook not found: {SOURCE_XLSX}")
    frame = pd.read_excel(
        SOURCE_XLSX,
        sheet_name="Plant by plant data",
        header=6,
        usecols="B:M",
        engine="openpyxl",
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    expected = {
        "Country", "City", "Project name", "Process type", "Subprocess type",
        "End-use", "Production capacity T/year", "Production capacity MWel",
        "Output T/year", "Latitude", "Longitude", "Year",
    }
    if set(frame.columns) != expected:
        raise ValueError(f"EHO source schema changed: {list(frame.columns)}")

    ammonia_all = frame.loc[
        frame["Country"].isin(EU_SAMPLE_COUNTRIES)
        & frame["End-use"].astype(str).str.strip().str.casefold().eq("ammonia")
    ].copy()
    electrolyser = ammonia_all.loc[
        ammonia_all["Process type"].astype(str).str.strip().str.casefold().eq("water electrolysis")
    ].copy()
    selected = ammonia_all.loc[
        ~ammonia_all["Process type"].astype(str).str.strip().str.casefold().eq("water electrolysis")
    ].copy()
    if selected.empty:
        raise ValueError("No non-electrolysis EU ammonia-use records found")

    numeric = [
        "Production capacity T/year", "Output T/year", "Latitude", "Longitude", "Year"
    ]
    for column in numeric:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna(
        subset=["Production capacity T/year", "Output T/year", "Latitude", "Longitude"]
    )
    if (selected["Output T/year"] <= 0).any() or selected["Year"].nunique() != 1:
        raise ValueError("Invalid output values or mixed reporting years in the selected sample")

    sites: list[dict[str, Any]] = []
    for (country, city), group in selected.groupby(["Country", "City"], sort=True, dropna=False):
        output = float(group["Output T/year"].sum())
        weights = group["Output T/year"].astype(float).to_numpy()
        latitudes = group["Latitude"].astype(float).to_numpy()
        longitudes = group["Longitude"].astype(float).to_numpy()
        project_names = sorted({str(value).strip() for value in group["Project name"] if pd.notna(value)})
        process_types = sorted({str(value).strip() for value in group["Process type"] if pd.notna(value)})
        city_text = str(city).strip()
        country_text = str(country).strip()
        sites.append({
            "site_id": f"{slug(country_text)}_{slug(city_text)}",
            "country": country_text,
            "timezone": TIMEZONES[country_text],
            "city_cluster": city_text,
            "project_names": " | ".join(project_names),
            "process_types": " | ".join(process_types),
            "eho_record_count": int(len(group)),
            "eho_year": int(group["Year"].iloc[0]),
            "h2_capacity_t_per_year": float(group["Production capacity T/year"].sum()),
            "h2_output_t_per_year_estimate": output,
            "estimated_utilization": output / float(group["Production capacity T/year"].sum()),
            "latitude_output_weighted": float((latitudes * weights).sum() / weights.sum()),
            "longitude_output_weighted": float((longitudes * weights).sum() / weights.sum()),
            "nh3_equivalent_service_tph": (
                output * 1000.0 / H2_PER_T_NH3_KG / 8760.0
            ),
            "source_capacity_unit": "t H2/year",
            "source_output_unit": "t H2/year; EHO estimate, not meter data",
        })
    result = pd.DataFrame(sites).sort_values(["country", "city_cluster"]).reset_index(drop=True)
    if result["site_id"].duplicated().any():
        raise ValueError("Site identifier collision after city clustering")

    metadata = {
        "source_page": EHO_PAGE,
        "source_download": EHO_XLSX_URL,
        "source_legal_notice": EHO_LEGAL_URL,
        "source_file": str(SOURCE_XLSX),
        "source_sha256": sha256_file(SOURCE_XLSX),
        "source_sheet": "Plant by plant data",
        "source_reporting_year": int(result["eho_year"].iloc[0]),
        "source_rows_five_covered_eu_countries_ammonia_use": int(len(ammonia_all)),
        "excluded_water_electrolysis_rows": int(len(electrolyser)),
        "included_non_electrolysis_rows": int(len(selected)),
        "included_city_clusters": int(len(result)),
        "included_countries": sorted(result["country"].unique().tolist()),
        "h2_capacity_t_per_year_sum": float(result["h2_capacity_t_per_year"].sum()),
        "h2_output_estimate_t_per_year_sum": float(result["h2_output_t_per_year_estimate"].sum()),
        "aggregation": "Rows sharing country and municipality are combined; coordinates are weighted by EHO-estimated H2 output.",
        "conversion": (
            "NH3-equivalent annual service = 2024 EHO estimated H2 output (t/y) "
            "* 1000 / 177.55300484072126 kg H2 per t NH3 / 8760 h/y."
        ),
        "interpretation_boundary": (
            "A coverage-limited set of public H2 production records with ammonia end-use, "
            "not a census of EU ammonia plants or plant operations. EHO output is estimated."
        ),
    }
    return result, metadata


def api_urls(site: dict[str, Any]) -> dict[str, str]:
    pv_params = {
        "lat": f"{site['latitude_output_weighted']:.8f}",
        "lon": f"{site['longitude_output_weighted']:.8f}",
        "startyear": 2005,
        "endyear": 2023,
        "outputformat": "json",
        "pvcalculation": 1,
        "peakpower": 1,
        "loss": 14,
        "trackingtype": 0,
        "angle": 35,
        "aspect": 0,
        "raddatabase": "PVGIS-ERA5",
    }
    wind_params = {
        "latitude": f"{site['latitude_output_weighted']:.8f}",
        "longitude": f"{site['longitude_output_weighted']:.8f}",
        "start_date": "2005-01-01",
        "end_date": "2023-12-31",
        "hourly": "wind_speed_100m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "models": "era5",
    }
    return {
        "pvgis": PVGIS_API + "?" + urllib.parse.urlencode(pv_params),
        "wind": OPEN_METEO_API + "?" + urllib.parse.urlencode(wind_params),
    }


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Project12-public-research/1.0 (reproducible academic use)",
            "Accept": "application/json",
        },
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {url}")
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 4:
                retry_after = exc.headers.get("Retry-After")
                time.sleep(max(15.0, float(retry_after) if retry_after else 15.0))
                continue
            raise
        except (
            TimeoutError,
            urllib.error.URLError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            ConnectionResetError,
        ):
            if attempt == 4:
                raise
            time.sleep(4.0 * (attempt + 1))
    raise RuntimeError(f"Unable to retrieve {url}")


def validate_weather(kind: str, content: bytes) -> dict[str, Any]:
    data = json.loads(content)
    if kind == "pvgis":
        hourly = data.get("outputs", {}).get("hourly", [])
        meteo = data.get("inputs", {}).get("meteo_data", {})
        if len(hourly) != EXPECTED_HOURS or meteo.get("meteo_db") != "ERA5":
            raise ValueError(f"PVGIS response failed coverage/source validation: {len(hourly)}; {meteo}")
        p = [float(row["P"]) for row in hourly]
        if any(value < 0 for value in p):
            raise ValueError("PVGIS response contains negative PV output")
        return {
            "records": len(hourly),
            "first_time": hourly[0]["time"],
            "last_time": hourly[-1]["time"],
            "meteo_database": meteo.get("meteo_db"),
            "power_unit": "W for a 1 kW fixed PV system; divide by 1000 for per-unit output",
        }
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    values = hourly.get("wind_speed_100m", [])
    if len(times) != EXPECTED_HOURS or len(values) != EXPECTED_HOURS:
        raise ValueError(f"Open-Meteo response failed coverage validation: {len(times)} times, {len(values)} values")
    if any(value is None or float(value) < 0 for value in values):
        raise ValueError("Open-Meteo response contains missing or negative wind speeds")
    return {
        "records": len(times),
        "first_time": times[0],
        "last_time": times[-1],
        "wind_speed_unit": hourly.get("wind_speed_100m_units"),
        "timezone": data.get("timezone"),
        "model_requested": "ERA5",
    }


def immutable_raw(path: Path, content: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256_file(path) != sha256_bytes(content):
            raise FileExistsError(f"Existing raw response differs; refusing overwrite: {path}")
        return True
    with NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(content)
        stream.flush()
    try:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite raw response: {path}")
        temp.rename(path)
    finally:
        if temp.exists():
            temp.unlink()
    return False


def acquire_weather(sample: pd.DataFrame, limit_sites: int | None) -> dict[str, Any]:
    selected = sample if limit_sites is None else sample.head(limit_sites)
    results: dict[str, Any] = {}
    for site in selected.to_dict(orient="records"):
        site_id = str(site["site_id"])
        urls = api_urls(site)
        results[site_id] = {}
        for kind, url in urls.items():
            subdir = "pvgis" if kind == "pvgis" else "open_meteo"
            path = RAW_ROOT / subdir / f"{site_id}_{kind}_era5_2005_2023.json"
            if path.exists():
                content = path.read_bytes()
                reused = True
            else:
                content = fetch(url)
                reused = immutable_raw(path, content)
            validation = validate_weather(kind, content)
            results[site_id][kind] = {
                "url": url,
                "raw_path": str(path),
                "bytes": len(content),
                "sha256": sha256_bytes(content),
                "reused_identical": reused,
                "validation": validation,
            }
            print(f"{site_id} {kind}: {validation['records']} hourly records", flush=True)
            time.sleep(1.0)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-weather", action="store_true")
    parser.add_argument("--limit-sites", type=int, default=None,
                        help="Acquire only the first N site clusters as an API smoke test")
    args = parser.parse_args()
    if args.limit_sites is not None and args.limit_sites < 1:
        parser.error("--limit-sites must be positive")

    sample, metadata = build_site_sample()
    manifest_path = MANIFEST_ROOT / (
        "acquisition_manifest_smoke_test.json"
        if args.limit_sites is not None
        else "acquisition_manifest.json"
    )
    sample_bytes = sample.to_csv(index=False, lineterminator="\n").encode("utf-8")
    write_derived_once(SAMPLE_PATH, sample_bytes)
    metadata.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_csv": str(SAMPLE_PATH),
        "sample_csv_sha256": sha256_bytes(sample_bytes),
        "weather_sources": {
            "pvgis_docs": PVGIS_DOC_URL,
            "pvgis_hourly_docs": PVGIS_HOURLY_DOC_URL,
            "open_meteo_docs": OPEN_METEO_DOC_URL,
            "period": "2005-01-01 through 2023-12-31",
            "hourly_records_per_series": EXPECTED_HOURS,
        },
        "weather_acquisition_complete": False,
        "weather_responses": {},
    })
    if args.download_weather:
        responses = acquire_weather(sample, args.limit_sites)
        metadata["weather_responses"] = responses
        metadata["weather_sites_acquired"] = len(responses)
        metadata["weather_acquisition_complete"] = len(responses) == len(sample)
        metadata["raw_root"] = str(RAW_ROOT)
    manifest_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
    write_derived_once(manifest_path, manifest_bytes)
    print(json.dumps({
        "sample_path": str(SAMPLE_PATH),
        "sample_rows": len(sample),
        "countries": metadata["included_countries"],
        "h2_output_estimate_t_per_year_sum": metadata["h2_output_estimate_t_per_year_sum"],
        "manifest_path": str(manifest_path),
        "weather_acquisition_complete": metadata["weather_acquisition_complete"],
        "weather_sites_acquired": metadata.get("weather_sites_acquired", 0),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
