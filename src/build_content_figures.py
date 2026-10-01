"""Four result figures, generated only from complete and audited current data."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'results' / 'eho_site_module_frontier' / 'primary_cyclic_v5_ds1_ml06_ml04_b24_holdout2015_2023'
FRONTIER = ROOT / 'results' / 'eho_threshold_sensitivity' / 'cyclic_continuous_module_frontier_v1'
RATE = ROOT / 'results' / 'eho_site_module_frontier' / 'declared_anchor_subgrid_v1'
OUT = ROOT / 'manuscript' / 'figures' / 'current_article'
CASES = ['rigid', 'flex_ml0p6_b24', 'flex_ml0p4_b24']
COLORS = ['#555555', '#237aab', '#c47820']
NAMES = ['Rigid', '60% floor + 24 h', '40% floor + 24 h']
ANCHORS = ['spain_sabinanigo', 'spain_huelva', 'netherlands_sluiskil']


def save(fig, name):
    fig.savefig(OUT / (name + '.svg'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.png'), dpi=400, bbox_inches='tight')
    plt.close(fig)


def main():
    for path, expected in [(MAIN/'validated_summary_flex_ml0p6_b24.json', 'validated_complete'),
                           (FRONTIER/'independent_frontier_audit.json', 'validated_complete'),
                           (RATE/'fixed_volume_deliverability_audit.json', 'complete')]:
        if json.loads(path.read_text())['status'] != expected:
            raise RuntimeError('Result figure requires completed validated data')
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    data = pd.read_csv(FRONTIER/'continuous_module_frontier.csv')
    fig, axes = plt.subplots(1,3,figsize=(9.5,3.8),sharey=True,constrained_layout=True)
    for ax, sid in zip(axes,ANCHORS):
        for j,case in enumerate(CASES):
            values=data[(data.site_id==sid)&(data.process_case==case)].sort_values('weather_year')
            ax.bar(np.arange(3)+(j-1)*.24, values.maximum_service_tph_per_reference_module,.23,color=COLORS[j],label=NAMES[j])
        ax.set_xticks(range(3),['2015','2019','2023']); ax.set_title(sid.split('_',1)[1].replace('_',' ').title())
        ax.set_xlabel('Weather year')
    axes[0].set_ylabel('Service capacity per reference unit\n(t NH3-equivalent / h)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    save(fig,'figure1_continuous_service_capacity')
    p=pd.read_csv(MAIN/'paired_site_weather_summary_flex_ml0p6_b24.csv')
    fig, axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    unchanged=p.reference_caverns_avoided.eq(0)
    for mask,label,color in [(unchanged,'Count unchanged',COLORS[0]),(~unchanged,'Lower count',COLORS[1])]:
        axes[0].scatter(p.loc[mask,'free_h2_storage_service_hours_rigid'],p.loc[mask,'free_h2_reduction_pct'],s=26,color=color,alpha=.7,label=label)
    axes[0].set_xlabel('Rigid free H2 inventory (service hours)'); axes[0].set_ylabel('Free H2 inventory reduction (%)'); axes[0].legend(frameon=False,fontsize=11)
    s=pd.read_csv(MAIN/'summary_by_site_flex_ml0p6_b24.csv').sort_values('site_id')
    y=np.arange(len(s))
    axes[1].scatter(s.all_replay_module_count_rigid,y,color=COLORS[0],label='Rigid')
    axes[1].scatter(s.all_replay_module_count_flex,y,color=COLORS[1],label='60% floor + 24 h')
    for yy,lo,hi in zip(y,s.all_replay_module_count_flex,s.all_replay_module_count_rigid): axes[1].plot([lo,hi],[yy,yy],color='#bbbbbb',zorder=0)
    axes[1].set_yticks(y,[v.split('_',1)[1].replace('_',' ').title() for v in s.site_id],fontsize=12)
    axes[1].set_xlabel('Modules covering all nine annual replays'); axes[1].legend(frameon=False,fontsize=11)
    save(fig,'figure2_inventory_and_design')
    b=pd.read_csv(MAIN/'capacity_deliverability_bounds.csv')
    primary=pd.read_csv(MAIN/'site_weather_frontier.csv')
    fig,ax=plt.subplots(figsize=(7,3.8),constrained_layout=True)
    for case,color,label in zip(CASES,COLORS,NAMES):
        bb=b[b.process_case==case]
        counts=bb.certified_deliverability_above_volume_bound.value_counts()
        ax.bar(label,int(counts.get(1,0)),color=color)
    ax.set_ylabel('Cases with a certified rate bound\nabove the free-inventory volume bound')
    ax.set_ylim(0,135); ax.set_title('Independent interval certificates: 135 cases per process')
    save(fig,'figure3_rate_certificates')
    r=pd.read_csv(RATE/'fixed_volume_deliverability_counterfactual.csv')
    if len(r)!=27 or not r.status.eq('optimal').all(): raise RuntimeError('Incomplete rate counterfactual')
    fig,axes=plt.subplots(1,3,figsize=(9.5,3.8),sharey=True,constrained_layout=True)
    for ax,sid in zip(axes,ANCHORS):
        for j,case in enumerate(CASES):
            values=r[(r.site_id==sid)&(r.process_case==case)].sort_values('weather_year')
            ax.bar(np.arange(3)+(j-1)*.24,values.minimum_withdrawal_kgph_per_normalized_tph,.23,color=COLORS[j],label=NAMES[j])
        ax.set_xticks(range(3),['2015','2019','2023']); ax.set_title(sid.split('_',1)[1].replace('_',' ').title()); ax.set_xlabel('Weather year')
    axes[0].set_ylabel('Minimum withdrawal capacity\n(kg H2 / h per normalized t NH3 / h)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    save(fig,'figure4_common_volume_deliverability')
    inputs=[FRONTIER/'continuous_module_frontier.csv',MAIN/'site_weather_frontier.csv',
            MAIN/'capacity_deliverability_bounds.csv',RATE/'fixed_volume_deliverability_counterfactual.csv']
    (OUT/'figure_manifest.json').write_text(json.dumps({'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    print('FOUR_CONTENT_FIGURES_COMPLETE',OUT)


if __name__=='__main__': main()
