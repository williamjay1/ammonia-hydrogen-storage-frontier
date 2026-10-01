"""Independently admit every declared review-added case and summarize pairs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'/'review_added_diagnostics_v1'


def paired(frame, endpoint):
    keys=['site_id','weather_year','factor_level']
    p=frame.pivot(index=keys,columns='process_case',values=endpoint)
    if p.isna().any().any(): raise RuntimeError('Unpaired diagnostic')
    effect=100*(p['flex_ml0p6_b24']/p['rigid']-1)
    if endpoint.startswith('minimum'): effect=-effect
    return effect


def describe(values):
    return {'pairs':len(values),'median_pct':float(np.median(values)),
            'iqr_pct':np.percentile(values,[25,75]).tolist(),
            'range_pct':[float(min(values)),float(max(values))],
            'positive_pairs':int((values>1e-6).sum())}


def main():
    report={'status':'validated_complete','blocks':{}}
    for block,levels,n in [('wind',[.25,.75],36),
                           ('volume',['low_investment_large_cavern','mid_investment_cavern'],36),
                           ('grid',[0.,.6,1.],18)]:
        manifest=json.loads((OUT/f'{block}_manifest.json').read_text())
        frame=pd.read_csv(OUT/f'{block}_diagnostics.csv')
        anchors=['spain_sabinanigo','spain_huelva','netherlands_sluiskil'] if block!='grid' else ['spain_huelva']
        expected={(s,y,l,p) for s in anchors for y in [2015,2019,2023] for l in levels
                  for p in ['rigid','flex_ml0p6_b24']}
        actual=set(map(tuple,frame[['site_id','weather_year','factor_level','process_case']].to_numpy()))
        if len(frame)!=n or frame.case_key.duplicated().any() or expected!=actual or manifest['status']!='complete':
            raise RuntimeError('Incomplete declared block: '+block)
        for path,sha in manifest['source_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha: raise RuntimeError('Changed input: '+path)
        optimal=frame[frame.status.eq('optimal')]
        residual='max_primal_residual' if block=='wind' else 'maximum_original_matrix_and_bound_residual'
        if optimal[residual].max()>1e-5: raise RuntimeError('Residual admission failure')
        item={'rows':n,'statuses':frame.status.value_counts().to_dict(),
              'maximum_residual':float(optimal[residual].max()),
              'csv_sha256':hashlib.sha256((OUT/f'{block}_diagnostics.csv').read_bytes()).hexdigest()}
        if block in ('wind','volume'):
            if len(optimal)!=n: raise RuntimeError('Non-optimal frontier case')
            endpoint='maximum_service_tph_per_reference_module' if block=='wind' else 'minimum_withdrawal_kgph_per_normalized_tph'
            effects=paired(frame,endpoint)
            item['by_level']={str(level):describe(effects.xs(level,level='factor_level')) for level in levels}
            keys=['site_id','weather_year','factor_level']
            unchanged=['pv_capacity_mw','wind_capacity_mw','electrolyser_mw','grid_connection_mw']
            if block=='volume': unchanged+=['fixed_working_gas_kg_per_normalized_tph']
            if (frame.groupby(keys)[unchanged].nunique()!=1).any().any(): raise RuntimeError('Mismatched pair capacities')
        else:
            if not frame[frame.factor_level.eq(0)].status.eq('infeasible').all(): raise RuntimeError('Unexpected zero-grid status')
            if not frame[frame.factor_level.gt(0)].status.eq('optimal').all(): raise RuntimeError('Unexpected positive-grid status')
            identity=float(optimal.combined_stock_identity_residual_kg.max())
            if identity>1e-7: raise RuntimeError('Stock identity failed')
            item['maximum_stock_identity_residual_kg']=identity
            item['by_level']={}
            for level in [.6,1.]:
                f=frame[frame.factor_level.eq(level)]
                p=f.pivot(index='weather_year',columns='process_case',values='free_h2_inventory_service_hours')
                relief=p.rigid-p.flex_ml0p6_b24
                flex=f[f.process_case.eq('flex_ml0p6_b24')]
                item['by_level'][str(level)]={'inventory_relief_service_hours_by_year':relief.to_dict(),
                    'all_relief_within_24h':bool(relief.le(24+1e-7).all()),
                    'same_electrolysis_rigid_reconstruction_feasible_pairs':int(flex.same_electrolysis_reconstruction_feasible.sum()),
                    'grid_capacity_mw_per_normalized_tph':float(f.grid_connection_mw.iloc[0])}
        report['blocks'][block]=item
    sample=pd.read_csv(ROOT/'datasets'/'eho_ammonia_site_clusters_v1_20260930.csv').sort_values('nh3_equivalent_service_tph')
    report['buffer_mass_tonnes']={r.site_id:24*r.nh3_equivalent_service_tph for _,r in sample.iloc[[0,len(sample)//2,-1]].iterrows()}
    report['scope']='All declared parametric cases, not observed clusters, a probability sample or plant/subsurface validation.'
    (OUT/'validated_diagnostic_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
