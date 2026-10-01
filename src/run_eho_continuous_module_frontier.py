"""Continuous volume-and-rate frontier, independent of EHO scale rounding.

u is working-gas allowance per normalized service, with rates in a fixed
reference ratio. One reference unit serves at most S_ref/u tonnes per hour.
For the homogeneous screen, ceil(D*u/S_ref) is the exact integer requirement.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, OptimizeWarning
import run_eho_site_module_frontier as site


def continuous_frontier(generation, months, *, capacities, q_floor, flexible,
                        buffer_hours, reference_volume, reference_withdrawal,
                        wtir=2., test_short=False):
    _, ub, b, eq, beq, bounds, idx = site.model.build_lp(
        generation, months, [len(generation)],
        electrolyser_mw=capacities['electrolyser_mw'],
        grid_connection_mw=capacities['grid_connection_mw'],
        min_hb_load_fraction=q_floor, flexible=flexible,
        nh3_buffer_limit_hours=buffer_hours, matching_rule='hourly',
        boundary='cyclic', hb_nameplate_tph=1.5,
        ramp_fraction_of_nameplate_per_hour=.6,
        withdrawal_to_injection_ratio=wtir,
        allow_short_periods_for_test=test_short)
    n, size = idx.hours, idx.size
    row = np.r_[0, 0, np.arange(1,n+1), np.arange(1,n+1),
                np.arange(n+1,2*n+1), np.arange(n+1,2*n+1)]
    col = np.r_[idx.h2_capacity, size,
                np.arange(idx.starts['h2_charge_kgph'],idx.starts['h2_charge_kgph']+n),
                np.full(n,size),
                np.arange(idx.starts['h2_discharge_kgph'],idx.starts['h2_discharge_kgph']+n),
                np.full(n,size)]
    ratio = reference_withdrawal/reference_volume
    val = np.r_[1.,-1.,np.ones(n),np.full(n,-ratio/wtir),np.ones(n),np.full(n,-ratio)]
    new = sparse.coo_matrix((val,(row,col)),shape=(2*n+1,size+1)).tocsr()
    ub = sparse.vstack([sparse.hstack([ub,sparse.csr_matrix((ub.shape[0],1))]),new],format='csr')
    b = np.r_[b,np.zeros(2*n+1)]
    eq = sparse.hstack([eq,sparse.csr_matrix((eq.shape[0],1))],format='csr')
    bounds = [*bounds,(0.,None)]
    objective=np.zeros(size+1)
    objective[-1]=1.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning)
        result=linprog(objective,A_ub=ub,b_ub=b,A_eq=eq,b_eq=beq,
                       bounds=bounds,method='highs-ds',options={'threads':1})
    if not result.success:
        raise RuntimeError(f'Continuous frontier solve failed: {result.message}')
    residual=max(float(np.max(np.abs(eq@result.x-beq))),
                 float(max(0,np.max(ub@result.x-b))))
    for value,(lower,upper) in zip(result.x,bounds):
        if lower is not None: residual=max(residual,lower-value)
        if upper is not None: residual=max(residual,value-upper)
    if residual>1e-5 or not np.isfinite(result.fun) or result.fun<=0:
        raise RuntimeError(f'Invalid frontier: objective={result.fun}, residual={residual}')
    return {'bundled_working_gas_allowance_kg_per_normalized_tph':float(result.fun),
            'maximum_service_tph_per_reference_module':float(reference_volume/result.fun),
            'max_primal_residual':residual,'status':'optimal'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-label',default='cyclic_continuous_module_frontier_v1')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    out=site.PROJECT/'results'/'eho_threshold_sensitivity'/args.run_label
    out.mkdir(parents=True,exist_ok=True)
    csv=out/'continuous_module_frontier.csv'
    manifest_path=out/'run_manifest.json'
    sample=pd.read_csv(site.SAMPLE_PATH).sort_values('nh3_equivalent_service_tph')
    anchors=[sample.iloc[0],sample.iloc[len(sample)//2],sample.iloc[-1]]
    case=next(x for x in site.model.CAVERN_CASES if x.case_id=='high_investment_small_cavern')
    dependencies=[Path(__file__),Path(site.__file__),Path(site.model.__file__),Path(site.weather.__file__),
                  Path(__import__('run_eu27_rfnbo_frontier_aux').__file__),
                  Path(__import__('rfnbo_inventory_frontier').__file__),site.SAMPLE_PATH]
    for point in anchors:
        dependencies += [site.RAW_ROOT/'pvgis'/f'{point.site_id}_pvgis_era5_2005_2023.json',
                         site.RAW_ROOT/'open_meteo'/f'{point.site_id}_wind_era5_2005_2023.json']
    hashes={str(p):digest(p) for p in dependencies}
    config={'years':[2015,2019,2023],'site_ids':[p.site_id for p in anchors],
            'process_cases':['rigid','flex_ml0p6_b24','flex_ml0p4_b24'],
            'reference':case.case_id,'wtir':2.,'boundary':'cyclic','solver':'highs-ds/threads=1'}
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if not args.resume or previous['configuration']!=config or previous['sha256']!=hashes:
            raise RuntimeError('Continuation requires identical sources and explicit --resume')
        rows=pd.read_csv(csv).to_dict('records') if csv.exists() else []
    else: rows=[]
    done={(r['site_id'],int(r['weather_year']),r['process_case']) for r in rows}
    def save(status):
        pd.DataFrame(rows).to_csv(csv,index=False)
        manifest_path.write_text(json.dumps({'status':status,'completed_rows':len(rows),
                'expected_rows':27,'configuration':config,'sha256':hashes},indent=2))
    save('running')
    for point in anchors:
        pv=site.pvgis_series(site.RAW_ROOT/'pvgis'/f'{point.site_id}_pvgis_era5_2005_2023.json')
        wind=site.wind_series(site.RAW_ROOT/'open_meteo'/f'{point.site_id}_wind_era5_2005_2023.json')
        capacities=site.model.design_capacities_with_aux(point.timezone,pv,wind,
                        site.model.AUX_MWH_PER_T_DEFAULT,renewable_overbuild=1.5,wind_energy_share=.5)
        for year in config['years']:
            _,generation,months,_,_=site.weather.profile_for_year(year,point.timezone,pv,wind,
                              capacities['pv_capacity_mw'],capacities['wind_capacity_mw'])
            for name,flex,floor,buffer in [('rigid',False,1.,0.),('flex_ml0p6_b24',True,.6,24.),
                                           ('flex_ml0p4_b24',True,.4,24.)]:
                if (point.site_id,year,name) in done: continue
                result=continuous_frontier(generation,months,capacities=capacities,
                            q_floor=floor,flexible=flex,buffer_hours=buffer,
                            reference_volume=case.working_gas_kg,
                            reference_withdrawal=case.peak_withdrawal_kg_per_hour)
                rows.append({'site_id':point.site_id,'weather_year':year,'process_case':name,
                             'eho_service_scale_tph':point.nh3_equivalent_service_tph,**result})
                done.add((point.site_id,year,name))
                save('running')
                print(point.site_id,year,name,result,flush=True)
    save('complete' if len(done)==27 else 'partial')


if __name__=='__main__':
    main()
