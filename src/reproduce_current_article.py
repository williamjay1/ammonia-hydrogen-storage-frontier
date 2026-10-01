"""Portable entry point for the supplied current-article code and inputs.

Only filesystem roles are redirected in memory. No frozen mathematical source
is edited. Fresh runs create new manifests and never resume the original run.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib
import json
import shutil
import sys
import numpy as np
import pandas as pd

PACKAGE=Path(__file__).resolve().parents[1]


def verify():
    manifest=json.loads((PACKAGE/'manifest_sha256.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Package checksum mismatch: '+name)
    weather=PACKAGE/'datasets'/'current_article_weather_cache'
    meta=json.loads((weather/'weather_cache_manifest.json').read_text())
    for item in meta['records']:
        data=pd.read_csv(weather/item['file'],float_precision='round_trip')
        if len(data)!=166536 or not np.isfinite(data[['pv_factor','wind_speed_100m_ms']]).all().all():
            raise RuntimeError('Invalid weather cache: '+item['site_id'])
    print('PACKAGE_INTEGRITY_AND_15_WEATHER_INPUTS_VERIFIED',flush=True)


def reconstructed_responses(work):
    raw=work/'cache'/'rehydrated_public_weather'
    weather=PACKAGE/'datasets'/'current_article_weather_cache'
    meta=json.loads((weather/'weather_cache_manifest.json').read_text())
    for item in meta['records']:
        data=pd.read_csv(weather/item['file'],float_precision='round_trip')
        starts=pd.to_datetime(data.utc_interval_start,utc=True)
        factors=data.pv_factor.to_numpy()
        powers=factors*1000.
        for direction in (np.inf,-np.inf):
            bad=powers/1000.!=factors
            candidate=np.nextafter(powers,direction)
            accept=bad & (candidate/1000.==factors)
            powers[accept]=candidate[accept]
        if not np.array_equal(powers/1000.,factors):
            raise RuntimeError('Cannot rehydrate the exact PV-factor floating-point values')
        pv={'inputs':{'meteo_data':{'meteo_db':'ERA5'}},'outputs':{'hourly':[
            {'time':stamp,'P':float(value)} for stamp,value in zip(
                (starts+pd.Timedelta(minutes=30)).dt.strftime('%Y%m%d:%H%M'),powers)]}}
        wind={'hourly':{'time':starts.dt.strftime('%Y-%m-%dT%H:%M').tolist(),
                        'wind_speed_100m':data.wind_speed_100m_ms.tolist()}}
        for sub,suffix,payload in [('pvgis','pvgis_era5_2005_2023',pv),('open_meteo','wind_era5_2005_2023',wind)]:
            target=raw/sub/f'{item["site_id"]}_{suffix}.json'
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): target.write_text(json.dumps(payload,separators=(',',':')))
    return raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['verify','primary','frontier','common-volume','threshold','figures','review-wind','review-volume','review-grid'],default='verify')
    parser.add_argument('--work-root',type=Path)
    parser.add_argument('--site-ids',nargs='+')
    parser.add_argument('--years',nargs='+',type=int)
    parser.add_argument('--anchor',choices=['low','median','high'])
    parser.add_argument('--max-new-cases',type=int)
    args=parser.parse_args()
    verify()
    if args.mode=='verify': return
    if args.work_root is None: parser.error('An explicit writable --work-root is required for computation')
    work=args.work_root.resolve()
    if sys.platform=='win32' and work.drive.upper()=='C:': parser.error('Do not write compute outputs to C:')
    work.mkdir(parents=True,exist_ok=True)
    if args.mode=='figures':
        figures=importlib.import_module('build_content_figures')
        figures.ROOT=PACKAGE
        figures.MAIN=PACKAGE/'results'/'primary'
        figures.FRONTIER=PACKAGE/'results'/'frontier'
        figures.RATE=PACKAGE/'results'/'common_volume'
        figures.OUT=work/'figures'
        figures.main()
        return
    site=importlib.import_module('run_eho_site_module_frontier')
    site.PROJECT=work
    site.OUT_ROOT=work/'results'/'eho_site_module_frontier'
    site.SAMPLE_PATH=PACKAGE/'datasets'/'eho_ammonia_site_clusters_v1_20260930.csv'
    site.RAW_ROOT=reconstructed_responses(work)
    site.weather.PROJECT=work
    site.weather.OUT_ROOT=work/'results'
    site.model.ROOT=work
    if args.mode.startswith('review-'):
        diagnostic=importlib.import_module('run_review_added_diagnostics')
        diagnostic.ROOT=work
        diagnostic.PRIMARY=PACKAGE/'results'/'primary'
        diagnostic.OUT=work/'results'/'review_added_diagnostics'
        sys.argv=['run_review_added_diagnostics.py','--block',args.mode.split('-',1)[1]]
        if args.max_new_cases: sys.argv+=['--max-new-cases',str(args.max_new_cases)]
        diagnostic.main()
    elif args.mode=='primary':
        sys.argv=['run_eho_site_module_frontier.py','--run-label','reproduced_primary',
                  '--min-load-fractions','.6','.4','--buffer-hours','24',
                  '--cavern-cases','high_investment_small_cavern','--wtir','2',
                  '--boundary','cyclic','--solver-method','highs-ds',
                  '--lp-formulation','expanded','--solver-threads','1']
        if args.site_ids: sys.argv+=['--site-ids',*args.site_ids]
        if args.years: sys.argv+=['--years',*map(str,args.years)]
        site.main()
    elif args.mode=='frontier':
        sys.argv=['run_eho_continuous_module_frontier.py','--run-label','reproduced_continuous_frontier']
        importlib.import_module('run_eho_continuous_module_frontier').main()
    elif args.mode=='common-volume':
        destination=work/'results'/'reproduced_common_volume'
        destination.mkdir(parents=True,exist_ok=False)
        for name in ('run_manifest.json','site_weather_frontier.csv'):
            shutil.copyfile(PACKAGE/'results'/'primary'/name,destination/name)
        sys.argv=['run_eho_unbundled_deliverability.py','--run-dir',str(destination)]
        importlib.import_module('run_eho_unbundled_deliverability').main()
    elif args.mode=='threshold':
        sys.argv=['run_eho_threshold_robustness.py','--run-label','reproduced_threshold_grid']
        if args.anchor: sys.argv+=['--anchor',args.anchor]
        importlib.import_module('run_eho_threshold_robustness').main()


if __name__=='__main__': main()
