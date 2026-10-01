# Data and estimand dictionary

All result rows are model outputs, not observations of plant operation.

| Field or quantity | Unit / interpretation |
| --- | --- |
| `site_id` | Public city-cluster scenario identifier, not a privately validated factory identifier |
| `weather_year` | One independently optimized full calendar-year weather replay |
| `process_case=rigid` | Constant synthesis service with zero product-buffer allowance |
| `process_case=flex_ml0p6_b24` | Minimum load 60% of a synthesis nameplate 1.5 times mean service; 24 mean-service-hour product buffer |
| `process_case=flex_ml0p4_b24` | Corresponding 40%-of-nameplate sensitivity case |
| `utc_interval_start` | Hourly UTC interval start, with leap years retained |
| `pv_factor` | Dimensionless PVGIS modeled output per installed kWp |
| `wind_speed_100m_ms` | ERA5 100-m wind speed in m/s, from Open-Meteo |
| EHO capacity/output columns | Public estimated annual hydrogen scale, t/year; not metered production or ammonia demand |
| `maximum_service_tph_per_reference_module` | Maximum normalized t NH3-equivalent/hour supported by one reference storage unit within the stated proportional non-storage sizing family |
| `minimum_withdrawal_kgph_per_normalized_tph` | Minimum kg H2/hour withdrawal rating per t NH3-equivalent/hour of constant service at a common working-gas volume |
| `free_h2_storage_service_hours_rigid` | Unconstrained hydrogen inventory divided by hourly stoichiometric service |
| `free_h2_reduction_pct` | Paired percentage reduction of the unconstrained hydrogen-inventory requirement |
| `optimized_reference_modules`, `reference_caverns_avoided` | Conceptual generic storage-unit count and paired difference; not actual caverns constructed or avoided |
| `certified_deliverability_above_volume_bound` | Independent necessary-condition certificate: withdrawal bound exceeds the working-gas-volume bound |
| `combined_necessary_module_lower_bound` | Necessary combined physical bound, not a sufficient feasibility test |
| `status` | Solver outcome; infeasible cases remain in the released scenario tables |

Figure 1 uses 27 continuous cases. Figure 2 uses all 135 primary matched
location–year pairs and their 15 cluster maxima. Figure 3 uses all 405 primary
case records, grouped into 135 configurations per process case. Figure 4 uses
27 common-volume withdrawal cases. Their exact source CSVs and SHA-256 hashes
are recorded by `src/build_nature_figures.py` in `figures/figure_layout_audit.json`.

The joint flexibility/buffer package is the primary comparison. A service
capacity gain is not a change in output at a fixed plant. Annual cyclic stocks
and perfect foresight test conditional weather/schedule adequacy, not an outage
probability or plant-reliability guarantee. See the manuscript for equations and
the full inferential boundaries.
