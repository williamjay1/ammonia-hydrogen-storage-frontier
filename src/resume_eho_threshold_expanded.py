"""Resume unchanged threshold cases with checked witnesses and expanded DS.

Preserves completed compact checkpoints with their original provenance.
Constructive witnesses are checked against every original LP row and bound.
"""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linprog, OptimizeWarning
import run_eho_threshold_robustness as original

site=original.site_model
BACKENDS=[]
RESIDUALS=[]


def residual(x,ub,b,eq,beq,bounds):
    value=max(float(np.max(np.abs(eq@x-beq))),float(max(0,np.max(ub@x-b))))
    for v,(lower,upper) in zip(x,bounds):
        if lower is not None: value=max(value,lower-v)
        if upper is not None: value=max(value,v-upper)
    return value


def feasible(generation,months,capacities,cavern,floor,wtir,buffer_hours,ramp=original.RAMP):
    c,ub,b,eq,beq,bounds,idx=site.model.build_lp(
        generation,months,[len(generation)],
        electrolyser_mw=capacities['electrolyser_mw'],
        grid_connection_mw=capacities['grid_connection_mw'],
        min_hb_load_fraction=floor,flexible=True,nh3_buffer_limit_hours=buffer_hours,
        matching_rule='hourly',boundary='cyclic',hb_nameplate_tph=1.5,
        ramp_fraction_of_nameplate_per_hour=ramp,cavern=cavern,cavern_count=1,
        withdrawal_to_injection_ratio=wtir)
    n=idx.hours
    h=site.model.H2_PER_NH3_KG_PER_T
    eta=site.model.H2_KG_PER_MWH_ELECTROLYSER
    maximum=np.minimum(generation,capacities['electrolyser_mw'])
    fraction=h*n/(eta*maximum.sum()) if maximum.sum()>0 else np.inf
    if fraction<=1:
        e=maximum*fraction
        net=eta*e-h
        charge=np.maximum(net,0.)
        discharge=np.maximum(-net,0.)
        stock=np.cumsum(net)
        stock-=stock.min()
        grid=np.maximum(e+site.model.AUX_MWH_PER_T_DEFAULT
                        +site.model.STORAGE_COMPRESSION_KWH_PER_KG/1000*charge-generation,0.)
        witness=np.zeros(idx.size)
        for field,values in [('electrolyser_mw',e),('nh3_tph',np.ones(n)),
                             ('h2_stock_kg',stock),('h2_charge_kgph',charge),
                             ('h2_discharge_kgph',discharge),('grid_import_mw',grid)]:
            witness[idx.starts[field]:idx.starts[field]+n]=values
        witness[idx.h2_capacity]=stock.max()
        error=residual(witness,ub,b,eq,beq,bounds)
        if error<=1e-5:
            BACKENDS.append('checked_constructive_rigid_witness')
            RESIDUALS.append(error)
            return True,'feasible_constructive_rigid_witness'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning)
        result=linprog(np.zeros_like(c),A_ub=ub,b_ub=b,A_eq=eq,b_eq=beq,
                       bounds=bounds,method='highs-ds',options={'threads':1})
    BACKENDS.append('expanded_highs_ds')
    if result.success:
        error=residual(result.x,ub,b,eq,beq,bounds)
        if error>1e-5: raise RuntimeError(f'Threshold residual {error}')
        RESIDUALS.append(error)
        return True,'optimal'
    if result.status==2: return False,'infeasible'
    raise RuntimeError(f'Threshold solver failed: {result.message}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor',choices=['low','median'],required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    base=site.PROJECT/'results'/'eho_threshold_sensitivity'
    seed=base/f'cyclic_compact_threshold_v2_{args.anchor}'
    destination=base/f'cyclic_expanded_witness_threshold_v3_{args.anchor}'
    destination.mkdir(parents=True,exist_ok=True)
    csv=destination/'load_threshold_robustness.csv'
    manifest_path=destination/'run_manifest.json'
    seed_manifest=json.loads((seed/'run_manifest.json').read_text())
    for name,digest in seed_manifest['input_sha256'].items():
        if original.sha256(Path(name))!=digest: raise RuntimeError(f'Seed input changed: {name}')
    for name,digest in seed_manifest['code_sha256'].items():
        if original.sha256(Path(name))!=digest: raise RuntimeError(f'Seed code changed: {name}')
    hashes={**seed_manifest['code_sha256'],str(Path(__file__).resolve()):original.sha256(Path(__file__))}
    source_provenance={'directory':str(seed),'manifest':seed_manifest,
                'manifest_sha256':original.sha256(seed/'run_manifest.json'),
                'csv_sha256':original.sha256(seed/'load_threshold_robustness.csv')}
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if not args.resume or previous['code_sha256']!=hashes:
            raise RuntimeError('Resume requires identical source hashes')
        rows=pd.read_csv(csv).to_dict('records')
        source_provenance=previous['seed_provenance']
    else:
        if args.resume: raise RuntimeError('No continuation exists')
        rows=pd.read_csv(seed/'load_threshold_robustness.csv').to_dict('records')
        for row in rows: row['evaluation_backend']='preserved_compact_checkpoint'
    done={(r['scenario'],r['site_id'],int(r['weather_year'])) for r in rows}
    config={**seed_manifest['configuration'],
            'solver':{'method':'highs-ds','formulation':'expanded','threads':1,
                      'witness':'complete original-matrix and bound-checked rigid schedule'}}
    def save(status):
        pd.DataFrame(rows).to_csv(csv,index=False)
        manifest_path.write_text(json.dumps({'status':status,'configuration':config,
            'completed_rows':len(rows),'expected_rows':30,'code_sha256':hashes,
            'input_sha256':seed_manifest['input_sha256'],'seed_provenance':source_provenance},indent=2))
    save('running')
    point=pd.read_csv(site.SAMPLE_PATH).set_index('site_id').loc[config['selected_site_ids'][0]]
    sid=point.name
    pv=site.pvgis_series(site.RAW_ROOT/'pvgis'/f'{sid}_pvgis_era5_2005_2023.json')
    wind=site.wind_series(site.RAW_ROOT/'open_meteo'/f'{sid}_wind_era5_2005_2023.json')
    original.feasible=feasible
    for year in original.YEARS:
        for scenario in original.SCENARIOS:
            if (scenario['scenario'],sid,year) in done: continue
            BACKENDS.clear(); RESIDUALS.clear()
            overbuild=float(scenario.get('renewable_overbuild',1.5))
            capacities=site.model.design_capacities_with_aux(point.timezone,pv,wind,
                site.model.AUX_MWH_PER_T_DEFAULT,renewable_overbuild=overbuild,wind_energy_share=.5)
            _,generation,months,_,_=site.weather.profile_for_year(year,point.timezone,pv,wind,
                 capacities['pv_capacity_mw'],capacities['wind_capacity_mw'])
            chosen=next(c for c in site.model.CAVERN_CASES
                        if c.case_id==scenario.get('cavern_id',original.CAVERN_ID))
            cavern=site.model.CavernCase(chosen.case_id,chosen.working_gas_million_sm3,
                scenario.get('withdrawal_rate_million_sm3_day',chosen.peak_withdrawal_million_sm3_day),
                chosen.reference_site_cavern_count,service_reference_nh3_tph=float(point.nh3_equivalent_service_tph))
            ramp=float(scenario.get('ramp_fraction_nameplate_per_hour',original.RAMP))
            result=original.threshold(generation,months,capacities,cavern,
                float(scenario['wtir']),float(scenario['buffer_hours']),ramp)
            row={'scenario':scenario['scenario'],'site_id':sid,'country':point.country,
                 'city_cluster':point.city_cluster,'scale_anchor':args.anchor,
                 'nh3_equivalent_service_tph':point.nh3_equivalent_service_tph,
                 'weather_year':year,'wtir':scenario['wtir'],
                 'product_buffer_cap_hours':scenario['buffer_hours'],
                 'cavern_id':cavern.case_id,
                 'withdrawal_rate_million_sm3_day':cavern.peak_withdrawal_million_sm3_day,
                 'ramp_fraction_nameplate_per_hour':ramp,'renewable_overbuild':overbuild,
                 **result,'evaluation_backend':';'.join(sorted(set(BACKENDS))),
                 'maximum_checked_feasible_primal_residual':max(RESIDUALS) if RESIDUALS else np.nan,
                 'interpretation':'Conditional one-reference-unit feasibility, not factory availability.'}
            rows.append(row); done.add((scenario['scenario'],sid,year)); save('running')
            print(sid,year,scenario['scenario'],result,flush=True)
    save('complete' if len(done)==30 else 'partial')


if __name__=='__main__':
    main()
