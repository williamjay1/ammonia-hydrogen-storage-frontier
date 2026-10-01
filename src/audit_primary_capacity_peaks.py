"""Audit all stored primary stock/flow peaks against declared reference ceilings."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import run_eho_site_module_frontier as site

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results'/'eho_site_module_frontier'/'primary_cyclic_v5_ds1_ml06_ml04_b24_holdout2015_2023'


def main():
    m=json.loads((RUN/'run_manifest.json').read_text())
    if m['status']!='complete': raise RuntimeError('Complete primary results required')
    frame=pd.read_csv(RUN/'site_weather_frontier.csv')
    if len(frame)!=405 or not frame.status.eq('optimal').all(): raise RuntimeError('Not all declared primary solves are optimal')
    ref=site.model.CAVERN_CASES[-1]
    n=frame.minimum_feasible_modules.to_numpy()
    d=frame.service_reference_nh3_tph.to_numpy()
    violations={
        'working_gas_kg_at_scale':frame.h2_storage_kg_at_site_scale.to_numpy()-n*ref.working_gas_kg,
        'withdrawal_kgph_at_scale':d*frame.max_h2_discharge_kgph_per_normalized_tph.to_numpy()-n*ref.peak_withdrawal_kg_per_hour,
        'injection_kgph_at_scale':d*frame.max_h2_charge_kgph_per_normalized_tph.to_numpy()-n*ref.peak_withdrawal_kg_per_hour/frame.withdrawal_to_injection_ratio.to_numpy(),
        'product_stock_t_per_normalized_tph':frame.peak_nh3_buffer_use_t_per_normalized_tph.to_numpy()-frame.nh3_buffer_limit_hours.to_numpy()}
    maxima={k:float(max(0.,np.max(v))) for k,v in violations.items()}
    if any(v>1e-5 for v in maxima.values()): raise RuntimeError(f'Capacity peak violation {maxima}')
    report={'status':'validated_complete','rows':405,'maximum_positive_ceiling_violations':maxima,
            'source_csv_sha256':hashlib.sha256((RUN/'site_weather_frontier.csv').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Independently verifies reported capacity/peak values, not unstored complete hourly schedules.'}
    (RUN/'capacity_peak_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
