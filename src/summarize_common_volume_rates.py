"""Validate and summarize the entire predeclared common-volume comparison."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results'/'eho_site_module_frontier'/'declared_anchor_subgrid_v1'


def main():
    audit=json.loads((RUN/'fixed_volume_deliverability_audit.json').read_text())
    if audit['status']!='complete': raise RuntimeError('Common-volume counterfactual incomplete')
    frame=pd.read_csv(RUN/'fixed_volume_deliverability_counterfactual.csv')
    keys=['site_id','weather_year','process_case']
    expected={(sid,y,case) for sid in ['spain_sabinanigo','spain_huelva','netherlands_sluiskil']
              for y in [2015,2019,2023] for case in ['rigid','flex_ml0p6_b24','flex_ml0p4_b24']}
    if len(frame)!=27 or frame.duplicated(keys).any() or set(map(tuple,frame[keys].to_numpy()))!=expected:
        raise RuntimeError('Not the complete declared rate grid')
    if not frame.status.eq('optimal').all() or frame.maximum_original_matrix_and_bound_residual.max()>1e-5:
        raise RuntimeError('Unverified rate solution')
    volume=frame.groupby(['site_id','weather_year']).fixed_working_gas_kg_per_normalized_tph.agg(['min','max'])
    if not np.allclose(volume['min'],volume['max']): raise RuntimeError('Volumes differ across treatment cases')
    base=frame[frame.process_case.eq('rigid')].set_index(['site_id','weather_year'])
    pairs=[]; summary={}
    for case in ['flex_ml0p6_b24','flex_ml0p4_b24']:
        flex=frame[frame.process_case.eq(case)].set_index(['site_id','weather_year'])
        p=base.add_suffix('_rigid').join(flex.add_suffix('_flex'))
        p['withdrawal_capacity_reduction_pct']=100*(1-p.minimum_withdrawal_kgph_per_normalized_tph_flex/p.minimum_withdrawal_kgph_per_normalized_tph_rigid)
        if p.withdrawal_capacity_reduction_pct.min()<-1e-5: raise RuntimeError('Feasible-set inclusion failed')
        p['flex_case']=case; pairs.append(p.reset_index())
        v=p.withdrawal_capacity_reduction_pct
        summary[case]={'median_withdrawal_capacity_reduction_pct':float(v.median()),
                      'range_pct':[float(v.min()),float(v.max())],
                      'iqr_pct':[float(v.quantile(.25)),float(v.quantile(.75))],
                      'strictly_positive_pairs':int((v>1e-5).sum())}
    pd.concat(pairs,ignore_index=True).to_csv(RUN/'paired_common_volume_rates.csv',index=False)
    out={'status':'validated_complete','rows':27,'paired_comparisons_per_flexible_case':9,
         'maximum_original_matrix_and_bound_residual':float(frame.maximum_original_matrix_and_bound_residual.max()),
         'by_process_case':summary,'same_volume_verified':True}
    (RUN/'validated_common_volume_summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__': main()
