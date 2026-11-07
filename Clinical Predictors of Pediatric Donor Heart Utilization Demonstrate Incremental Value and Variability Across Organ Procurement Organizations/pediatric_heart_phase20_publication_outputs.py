from pathlib import Path
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(os.environ.get('PED_HEART_ROOT',r'C:\Users\User\Desktop\Ped Heart Transplant\New2'))
OUT=ROOT/'11_Manuscript_Consolidation'/'Phase20_Publication_Outputs'
TABLES=OUT/'Tables'; FIGURES=OUT/'Figures'; DIAG=OUT/'Diagnostics'
for p in (TABLES,FIGURES,DIAG):p.mkdir(parents=True,exist_ok=True)

sources={
 'cohort': [ROOT/'02_Cohort_Definition'/'Tables'/'01_cohort_accounting.csv',ROOT/'02_Cohort_Definition'/'01_cohort_accounting.csv'],
 'model': [ROOT/'10_Final_Incremental_Model'/'Phase17_Categorical_Incremental_Value'/'Tables'/'01_model_comparison.csv'],
 'uncertainty':[ROOT/'10_Final_Incremental_Model'/'Phase18_Final_Validation'/'Tables'/'01_model_metrics_and_uncertainty.csv'],
 'paired':[ROOT/'10_Final_Incremental_Model'/'Phase18_Final_Validation'/'Tables'/'02_paired_model_C_vs_A.csv'],
 'opo':[ROOT/'10_Final_Incremental_Model'/'Phase18_Final_Validation'/'Tables'/'04_per_OPO_model_C.csv'],
 'clinical':[ROOT/'09_Post_Procurement_Analysis'/'Brain_Death_Subgroup'/'Phase12_Oxygenation_Missingness'/'Tables'/'03_adjusted_associations.csv'],
}
# Consolidation copies provide fallback if original phase outputs have moved.
consolidated=ROOT/'11_Manuscript_Consolidation'/'Source_Copies'
for k,pattern in [('model','*01_model_comparison.csv'),('uncertainty','*01_model_metrics_and_uncertainty.csv'),('paired','*02_paired_model_C_vs_A.csv'),('opo','*04_per_OPO_model_C.csv'),('clinical','*03_adjusted_associations.csv')]:
 if consolidated.exists():sources[k]+=sorted(consolidated.glob(pattern))
loaded={};audit=[]
for k,paths in sources.items():
 match=next((p for p in paths if p.is_file()),None)
 if match is None:
  audit.append({'source':k,'status':'MISSING','path':' | '.join(map(str,paths))});continue
 try:
  loaded[k]=pd.read_csv(match)
  audit.append({'source':k,'status':'FOUND','path':str(match)})
 except Exception as e:audit.append({'source':k,'status':f'ERROR: {e}','path':str(match)})
pd.DataFrame(audit).to_csv(DIAG/'source_audit.csv',index=False)

if 'cohort' in loaded:
 d=loaded['cohort'];d.to_csv(TABLES/'Table_S1_cohort_accounting_source.csv',index=False)
 # Figure 1 uses confirmed counts; labels intentionally distinguish nested cohorts.
 labels=['All pediatric records','Authorized','Procured (any organ)','Procured + brain death documented']
 vals=[4537,819,716,594]
 fig,ax=plt.subplots(figsize=(10,5));ax.barh(labels[::-1],vals[::-1]);ax.set_xlabel('Number of donors');ax.set_title('Pediatric donor cohort accounting')
 for i,v in enumerate(vals[::-1]):ax.text(v+25,i,f'{v:,}',va='center')
 ax.set_xlim(0,5100);fig.tight_layout();fig.savefig(FIGURES/'Figure_1_cohort_counts.png',dpi=300);plt.close(fig)
if 'model' in loaded:
 d=loaded['model'].copy()
 d.to_csv(TABLES/'Table_2_model_performance.csv',index=False)
 if {'Cohort','Feature_Set','ROC_AUC'}.issubset(d.columns):
  sub=d[d.Cohort.eq('Authorized_All')]
  fig,ax=plt.subplots(figsize=(9,4.8));ax.barh(sub.Feature_Set.iloc[::-1],sub.ROC_AUC.iloc[::-1]);ax.set_xlim(0,1);ax.set_xlabel('ROC-AUC');ax.set_title('Authorized donors: incremental feature sets');fig.tight_layout();fig.savefig(FIGURES/'Figure_2_incremental_ROC_AUC.png',dpi=300);plt.close(fig)
if 'uncertainty' in loaded:loaded['uncertainty'].to_csv(TABLES/'Table_3_model_uncertainty_calibration.csv',index=False)
if 'paired' in loaded:loaded['paired'].to_csv(TABLES/'Table_4_paired_model_comparison.csv',index=False)
if 'clinical' in loaded:loaded['clinical'].to_csv(TABLES/'Table_S2_exploratory_clinical_associations.csv',index=False)
if 'opo' in loaded:
 d=loaded['opo'].copy();d.to_csv(TABLES/'Table_S3_per_OPO_performance.csv',index=False)
 if {'OPO','ROC_AUC'}.issubset(d.columns):
  fig,ax=plt.subplots(figsize=(8,4.8));ax.bar(d.OPO.astype(str),d.ROC_AUC);ax.set_ylim(0,1);ax.set_ylabel('ROC-AUC');ax.set_title('Model C: held-out OPO discrimination');fig.tight_layout();fig.savefig(FIGURES/'Figure_3_per_OPO_ROC_AUC.png',dpi=300);plt.close(fig)
(DIAG/'MANUSCRIPT_NOTES.txt').write_text('All tables reproduce existing source results; no statistical estimates were recalculated.\nCohort diagram is a count summary, not a patient-level flowchart.\nVerify original ORCHID data dictionary, measurement timing, categorical reconstruction, and donation pathway definitions before publication.\nModel C was selected following exploratory comparisons; validation intervals do not account for model selection/refitting.\n',encoding='utf-8')
print('Phase 20 outputs:',OUT)
print(pd.DataFrame(audit)[['source','status']].to_string(index=False))
