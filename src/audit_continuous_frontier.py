"""Independently check scale homogeneity and explain continuous rate plateaus."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import audit_eho_capacity_deliverability as cuts
import run_eho_site_module_frontier as site

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results' / 'eho_threshold_sensitivity' / 'cyclic_continuous_module_frontier_v1'
PRIMARY = ROOT / 'results' / 'eho_site_module_frontier' / 'declared_anchor_subgrid_v1'


def main():
    primary = pd.read_csv(PRIMARY / 'site_weather_frontier.csv').set_index(['site_id', 'weather_year', 'process_case'])
    frontier = pd.read_csv(RUN / 'continuous_module_frontier.csv')
    sample = pd.read_csv(site.SAMPLE_PATH).set_index('site_id')
    ref = site.model.CAVERN_CASES[-1]
    rows = []
    for sid, group in frontier.groupby('site_id'):
        point = sample.loc[sid]
        pv = site.pvgis_series(site.RAW_ROOT / 'pvgis' / f'{sid}_pvgis_era5_2005_2023.json')
        wind = site.wind_series(site.RAW_ROOT / 'open_meteo' / f'{sid}_wind_era5_2005_2023.json')
        for row in group.to_dict('records'):
            p = primary.loc[(sid, row['weather_year'], row['process_case'])]
            _, generation, _, _, _ = site.weather.profile_for_year(row['weather_year'], point.timezone, pv, wind,
                   p.pv_capacity_mw_per_normalized_tph, p.wind_capacity_mw_per_normalized_tph)
            cap = cuts.H2_PER_MWH * np.minimum(generation, p.electrolyser_capacity_mw_per_normalized_tph)
            qmin = 1. if row['process_case'] == 'rigid' else p.min_hb_load_fraction_of_nameplate * p.hb_nameplate_multiple
            limits = cuts.service_rate_bounds(cap, minimum_nh3_tph=qmin,
                product_buffer_hours=p.nh3_buffer_limit_hours, service_scale_tph=1.,
                working_gas_kg=ref.working_gas_kg, withdrawal_kgph=ref.peak_withdrawal_kg_per_hour,
                wtir=2., free_inventory_kg_per_tph=p.free_h2_storage_kg_per_normalized_tph)
            lower = max(p.free_h2_storage_kg_per_normalized_tph / ref.working_gas_kg,
                        limits['volume_cut_continuous_ratio'], limits['withdrawal_cut_continuous_ratio'],
                        limits['recharge_cut_continuous_ratio'])
            ub = 1./lower
            dmax = row['maximum_service_tph_per_reference_module']
            predicted = int(np.ceil(p.service_reference_nh3_tph / dmax - 1e-8))
            if predicted != int(p.minimum_feasible_modules) or dmax > ub + 1e-5:
                raise RuntimeError(f'Independent frontier consistency failed: {sid}/{row["weather_year"]}')
            deficit_one_hour = max(0., cuts.H2_PER_T*qmin - cap.min())
            instantaneous = ref.peak_withdrawal_kg_per_hour / deficit_one_hour if deficit_one_hour else np.inf
            rows.append({'site_id':sid, 'weather_year':row['weather_year'], 'process_case':row['process_case'],
                         'continuous_maximum_service_tph':dmax, 'interval_upper_bound_service_tph':ub,
                         'interval_bound_gap_pct':100*(ub/dmax-1),
                         'instantaneous_withdrawal_ceiling_tph':instantaneous,
                         'instantaneous_ceiling_attained':bool(abs(dmax-instantaneous)<1e-5),
                         'minimum_eligible_h2_kgph_per_normalized_tph':float(cap.min()),
                         'predicted_integer_count':predicted, 'independent_integer_count':int(p.minimum_feasible_modules),
                         'distance_of_scaled_continuous_count_to_integer':abs(p.service_reference_nh3_tph/dmax-round(p.service_reference_nh3_tph/dmax))})
    data = pd.DataFrame(rows)
    data.to_csv(RUN / 'independent_continuous_frontier_checks.csv',index=False)
    report={'status':'validated_complete','rows':len(data),'all_integer_counts_match':True,
            'all_interval_bounds_satisfied':True,'instantaneous_withdrawal_ceiling_attained_cases':int(data.instantaneous_ceiling_attained.sum()),
            'by_process_case':data.groupby('process_case').instantaneous_ceiling_attained.sum().astype(int).to_dict(),
            'minimum_distance_of_scaled_count_to_integer':float(data.distance_of_scaled_continuous_count_to_integer.min()),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_csv_sha256':hashlib.sha256((RUN/'continuous_module_frontier.csv').read_bytes()).hexdigest(),
            'primary_csv_sha256':hashlib.sha256((PRIMARY/'site_weather_frontier.csv').read_bytes()).hexdigest()}
    (RUN / 'independent_frontier_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
