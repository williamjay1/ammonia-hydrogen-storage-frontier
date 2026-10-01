"""Audit the completed continuous service-scale frontier and its scale scenarios."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results'/'eho_threshold_sensitivity'/'cyclic_continuous_module_frontier_v1'


def main():
    manifest=json.loads((RUN/'run_manifest.json').read_text())
    data=pd.read_csv(RUN/'continuous_module_frontier.csv')
    config=manifest['configuration']
    keys=['site_id','weather_year','process_case']
    expected={(sid,year,case) for sid in config['site_ids'] for year in config['years']
              for case in config['process_cases']}
    if (manifest['status']!='complete' or len(data)!=27 or data.duplicated(keys).any()
            or set(map(tuple,data[keys].to_numpy()))!=expected):
        raise RuntimeError('Continuous frontier is not the complete declared grid')
    if not data.status.eq('optimal').all() or data.max_primal_residual.max()>1e-5:
        raise RuntimeError('Unverified frontier solution')
    base=data[data.process_case.eq('rigid')].set_index(['site_id','weather_year'])
    pairs=[]
    for case in ['flex_ml0p6_b24','flex_ml0p4_b24']:
        flex=data[data.process_case.eq(case)].set_index(['site_id','weather_year'])
        joined=base.add_suffix('_rigid').join(flex.add_suffix('_flex'))
        joined['flex_case']=case
        joined['service_capacity_gain_pct']=100*(joined.maximum_service_tph_per_reference_module_flex
                                    /joined.maximum_service_tph_per_reference_module_rigid-1)
        joined['bundled_allowance_reduction_pct']=100*(1-
            joined.bundled_working_gas_allowance_kg_per_normalized_tph_flex
            /joined.bundled_working_gas_allowance_kg_per_normalized_tph_rigid)
        if joined.service_capacity_gain_pct.min() < -1e-5:
            raise RuntimeError('Feasible-set inclusion failed')
        pairs.append(joined.reset_index())
    pairs=pd.concat(pairs,ignore_index=True)
    pairs.to_csv(RUN/'paired_continuous_frontier.csv',index=False)
    rows=[]
    for row in data.to_dict('records'):
        for factor in [.5,.75,1.,1.25,1.5]:
            scale=factor*row['eho_service_scale_tph']
            ratio=scale/row['maximum_service_tph_per_reference_module']
            rows.append({'site_id':row['site_id'],'weather_year':row['weather_year'],
                         'process_case':row['process_case'],'eho_scale_factor':factor,
                         'scenario_service_scale_tph':scale,
                         'continuous_reference_module_equivalents':ratio,
                         'integer_reference_modules':int(np.ceil(ratio-1e-8)),
                         'distance_to_nearest_integer':abs(ratio-round(ratio))})
    pd.DataFrame(rows).to_csv(RUN/'service_scale_scenarios.csv',index=False)
    robust=data.groupby(['site_id','process_case'],as_index=False).agg(
        service_capacity_covering_three_independent_replays=('maximum_service_tph_per_reference_module','min'))
    robust.to_csv(RUN/'three_year_capacity_by_location.csv',index=False)
    summary={'status':'validated_complete','rows':27,'matched_anchor_year_pairs_per_comparison':9,
             'maximum_primal_residual':float(data.max_primal_residual.max()),'by_flex_case':{},
             'source_csv_sha256':hashlib.sha256((RUN/'continuous_module_frontier.csv').read_bytes()).hexdigest(),
             'interpretation':'Capacity gain of a generic bundled volume-and-rate allowance under enumerated weather; no factory availability or EU population inference.'}
    for case,group in pairs.groupby('flex_case'):
        values=group.service_capacity_gain_pct
        summary['by_flex_case'][case]={'median_service_capacity_gain_pct':float(values.median()),
            'range_service_capacity_gain_pct':[float(values.min()),float(values.max())],
            'iqr_service_capacity_gain_pct':[float(values.quantile(.25)),float(values.quantile(.75))],
            'median_bundled_allowance_reduction_pct':float(group.bundled_allowance_reduction_pct.median()),
            'strictly_positive_pairs':int((values>1e-5).sum())}
    (RUN/'validated_continuous_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
