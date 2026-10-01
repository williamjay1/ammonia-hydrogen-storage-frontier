"""Review-added one-factor diagnostics using unchanged public inputs and model."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
import run_eho_site_module_frontier as site
from run_eho_continuous_module_frontier import continuous_frontier
from run_eho_unbundled_deliverability import minimum_deliverability
from compact_cavern_lp import solve_compact

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT/'results'/'eho_site_module_frontier'/'primary_cyclic_v5_ds1_ml06_ml04_b24_holdout2015_2023'
OUT = ROOT/'results'/'review_added_diagnostics_v1'
CORE_HASH = '74176786c74102dac44eed87b31617053304b1ecd4f0c146daff443176df61f6'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid_solution(generation, months, capacities, *, flexible):
    alpha = site.model.AUX_MWH_PER_T_DEFAULT
    eta = site.model.H2_KG_PER_MWH_ELECTROLYSER
    h = site.model.H2_PER_NH3_KG_PER_T
    gamma = site.model.STORAGE_COMPRESSION_KWH_PER_KG/1000
    c, ub, b, eq, beq, bounds, idx = site.model.build_lp(
        generation, months, [len(generation)],
        electrolyser_mw=capacities['electrolyser_mw'],
        grid_connection_mw=capacities['grid_connection_mw'],
        min_hb_load_fraction=.6 if flexible else 1., flexible=flexible,
        nh3_buffer_limit_hours=24. if flexible else 0., matching_rule='hourly',
        boundary='cyclic', hb_nameplate_tph=1.5,
        ramp_fraction_of_nameplate_per_hour=.6)
    solution = solve_compact(c,A_ub=ub,b_ub=b,A_eq=eq,b_eq=beq,
                             bounds=bounds,method='highs-ipm',options={'threads':1})
    if solution.status == 2:
        return {'status':'infeasible','solver_status':2}
    if not solution.success:
        raise RuntimeError(solution.message)
    residual = max(float(np.max(np.abs(eq@solution.x-beq))),float(max(0,np.max(ub@solution.x-b))))
    for value,(lower,upper) in zip(solution.x,bounds):
        if lower is not None: residual=max(residual,lower-value)
        if upper is not None: residual=max(residual,value-upper)
    if residual > 1e-5: raise RuntimeError(f'Grid diagnostic residual: {residual}')
    n = idx.hours
    def block(name): return solution.x[idx.starts[name]:idx.starts[name]+n]
    e, s, a = block('electrolyser_mw'), block('h2_stock_kg'), block('nh3_stock_t')
    x = s+h*a
    identity = float(np.max(np.abs(x-np.roll(x,1)-eta*e+h)))
    reconstructed_grid = np.maximum(0,e+alpha+gamma*np.maximum(eta*e-h,0)-generation)
    requirement = float(np.max(reconstructed_grid))
    return {'status':'optimal','solver_status':0,
            'free_h2_inventory_kg_per_normalized_tph':float(solution.fun),
            'free_h2_inventory_service_hours':float(solution.fun/h),
            'maximum_original_matrix_and_bound_residual':residual,
            'combined_stock_identity_residual_kg':identity,
            'same_electrolysis_rigid_reconstruction_grid_requirement_mw':requirement,
            'same_electrolysis_reconstruction_feasible':requirement<=capacities['grid_connection_mw']+1e-7}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block',choices=['wind','volume','grid'],required=True)
    p.add_argument('--max-new-cases',type=int)
    args=p.parse_args()
    if digest(site.model.__file__) != CORE_HASH: raise RuntimeError('Frozen model has changed')
    sample=pd.read_csv(site.SAMPLE_PATH).sort_values('nh3_equivalent_service_tph')
    anchors=[sample.iloc[0],sample.iloc[len(sample)//2],sample.iloc[-1]]
    primary=pd.read_csv(PRIMARY/'site_weather_frontier.csv')
    small=site.model.CAVERN_CASES[-1]
    files=[Path(__file__),Path(site.model.__file__),Path(site.__file__),
           Path(__import__('compact_cavern_lp').__file__),
           Path(__import__('run_eho_continuous_module_frontier').__file__),
           Path(__import__('run_eho_unbundled_deliverability').__file__),
           site.SAMPLE_PATH,PRIMARY/'site_weather_frontier.csv']
    for point in anchors:
        files += [site.RAW_ROOT/'pvgis'/f'{point.site_id}_pvgis_era5_2005_2023.json',
                  site.RAW_ROOT/'open_meteo'/f'{point.site_id}_wind_era5_2005_2023.json']
    hashes={str(path):digest(path) for path in files}
    OUT.mkdir(parents=True,exist_ok=True)
    csv=OUT/f'{args.block}_diagnostics.csv'
    manifest=OUT/f'{args.block}_manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        if old['source_hashes'] != hashes: raise RuntimeError('Changed diagnostic source/input: use a new run version')
    rows=pd.read_csv(csv).to_dict('records') if csv.exists() else []
    done={r['case_key'] for r in rows}
    expected=18 if args.block=='grid' else 36
    def save():
        pd.DataFrame(rows).to_csv(csv,index=False)
        manifest.write_text(json.dumps({'status':'complete' if len(rows)==expected else 'running',
            'block':args.block,'expected_cases':expected,'completed_cases':len(rows),
            'source_hashes':hashes,'main_model_unchanged':True,
            'scope':'Review-added parametric public-weather scenarios, not new observed clusters or a probability sample'},indent=2),encoding='utf-8')
    new_cases=0
    for point in anchors:
        if args.block=='grid' and point.site_id!=anchors[1].site_id: continue
        pv=site.pvgis_series(site.RAW_ROOT/'pvgis'/f'{point.site_id}_pvgis_era5_2005_2023.json')
        wind=site.wind_series(site.RAW_ROOT/'open_meteo'/f'{point.site_id}_wind_era5_2005_2023.json')
        levels = [.25,.75] if args.block=='wind' else ([case.case_id for case in site.model.CAVERN_CASES[:-1]] if args.block=='volume' else [0.,.6,1.])
        for level in levels:
            caps=site.model.design_capacities_with_aux(point.timezone,pv,wind,
                site.model.AUX_MWH_PER_T_DEFAULT,renewable_overbuild=1.5,
                wind_energy_share=float(level) if args.block=='wind' else .5)
            grid_crit=site.model.AUX_MWH_PER_T_DEFAULT+site.model.STORAGE_COMPRESSION_KWH_PER_KG/1000*site.model.H2_KG_PER_MWH_ELECTROLYSER*caps['electrolyser_mw']
            if args.block=='grid': caps['grid_connection_mw']=float(level)*grid_crit
            reference=next(case for case in site.model.CAVERN_CASES if case.case_id==level) if args.block=='volume' else small
            for year in (2015,2019,2023):
                _,gen,months,_,_=site.weather.profile_for_year(year,point.timezone,pv,wind,caps['pv_capacity_mw'],caps['wind_capacity_mw'])
                rigid=primary[(primary.site_id==point.site_id)&(primary.weather_year==year)&(primary.process_case=='rigid')].iloc[0]
                units=max(1,int(math.ceil(point.nh3_equivalent_service_tph*rigid.free_h2_storage_kg_per_normalized_tph/reference.working_gas_kg-1e-9)))
                volume=units*reference.working_gas_kg/point.nh3_equivalent_service_tph
                for name,flex,floor,buffer in [('rigid',False,1.,0.),('flex_ml0p6_b24',True,.6,24.)]:
                    key=f'{args.block}|{point.site_id}|{level}|{year}|{name}'
                    if key in done: continue
                    started=time.perf_counter()
                    if args.block=='wind':
                        result=continuous_frontier(gen,months,capacities=caps,q_floor=floor,flexible=flex,
                            buffer_hours=buffer,reference_volume=small.working_gas_kg,reference_withdrawal=small.peak_withdrawal_kg_per_hour)
                    elif args.block=='volume':
                        result=minimum_deliverability(gen,months,capacities=caps,q_floor=floor,flexible=flex,
                            buffer_hours=buffer,fixed_volume=volume)
                    else: result=grid_solution(gen,months,caps,flexible=flex)
                    record={'case_key':key,'block':args.block,'site_id':point.site_id,'weather_year':year,
                            'factor_level':level,'process_case':name,'service_scale_tph':point.nh3_equivalent_service_tph,
                            'grid_critical_sufficient_mw':grid_crit,'grid_connection_mw':caps['grid_connection_mw'],
                            'pv_capacity_mw':caps['pv_capacity_mw'],'wind_capacity_mw':caps['wind_capacity_mw'],
                            'electrolyser_mw':caps['electrolyser_mw'],
                            'elapsed_seconds':time.perf_counter()-started,**result}
                    rows.append(record);done.add(key);new_cases+=1;save()
                    print(json.dumps({'case':key,'completed':len(rows),'seconds':record['elapsed_seconds'],**result}),flush=True)
                    if args.max_new_cases and new_cases>=args.max_new_cases: return
    if len(rows)!=expected: raise RuntimeError('Declared grid incomplete')
    save()


if __name__=='__main__': main()
