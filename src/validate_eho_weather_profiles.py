"""Validate immutable hourly PVGIS/Open-Meteo inputs for the EHO cluster sample."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from acquire_eho_ammonia_site_weather import PROJECT, RAW_ROOT, SAMPLE_PATH
from run_eho_site_module_frontier import pvgis_series, wind_series


OUT_PATH = PROJECT / "data_manifest" / "eho_ammonia_sites_2024_v1" / "weather_profile_validation_20260930.json"
EXPECTED_HOURS = 166_536


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> dict[str, object]:
    sample = pd.read_csv(SAMPLE_PATH)
    records: list[dict[str, object]] = []
    for site in sample.to_dict(orient="records"):
        site_id = str(site["site_id"])
        pv_path = RAW_ROOT / "pvgis" / f"{site_id}_pvgis_era5_2005_2023.json"
        wind_path = RAW_ROOT / "open_meteo" / f"{site_id}_wind_era5_2005_2023.json"
        pv = pvgis_series(pv_path)
        wind = wind_series(wind_path)
        if len(pv) != EXPECTED_HOURS or len(wind) != EXPECTED_HOURS:
            raise ValueError(f"Unexpected hourly sample size for {site_id}")
        if not pv.index.equals(wind.index):
            raise ValueError(f"PV/wind timestamps do not align for {site_id}")
        if pv.index[0] != pd.Timestamp("2005-01-01 00:00:00+00:00"):
            raise ValueError(f"Unexpected first UTC interval for {site_id}")
        if pv.index[-1] != pd.Timestamp("2023-12-31 23:00:00+00:00"):
            raise ValueError(f"Unexpected last UTC interval for {site_id}")
        records.append({
            "site_id": site_id,
            "records_per_source": EXPECTED_HOURS,
            "first_interval_utc": pv.index[0].isoformat(),
            "last_interval_utc": pv.index[-1].isoformat(),
            "pvgis_raw_bytes": pv_path.stat().st_size,
            "open_meteo_raw_bytes": wind_path.stat().st_size,
            "pvgis_sha256": file_hash(pv_path),
            "open_meteo_sha256": file_hash(wind_path),
            "pv_min_max_pu": [float(pv.min()), float(pv.max())],
            "wind_min_max_m_s": [float(wind.min()), float(wind.max())],
        })
    return {
        "status": "passed",
        "sample_path": str(SAMPLE_PATH),
        "sample_rows": len(sample),
        "sites_validated": len(records),
        "years": [2005, 2023],
        "expected_records_per_site_per_source": EXPECTED_HOURS,
        "time_basis": "UTC hourly interval starts; PVGIS :30 means shifted back by 30 minutes",
        "records": records,
    }


if __name__ == "__main__":
    result = validate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if OUT_PATH.exists() and OUT_PATH.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"Refusing to replace differing validation manifest: {OUT_PATH}")
    if not OUT_PATH.exists():
        OUT_PATH.write_text(payload, encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False, indent=2))
