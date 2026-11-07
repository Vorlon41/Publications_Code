from pathlib import Path
import shutil, json, csv, datetime

ROOT = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT = ROOT / '11_Manuscript_Consolidation'
for sub in ['Tables','Figures','Diagnostics','Source_Copies']:
    (OUT/sub).mkdir(parents=True,exist_ok=True)

sources = [
 ('Cohort accounting', '02_Cohort_Definition', '01_cohort_accounting.csv'),
 ('Phase 17 model comparison','10_Final_Incremental_Model/Phase17_Categorical_Incremental_Value/Tables','01_model_comparison.csv'),
 ('Phase 17 paired comparisons','10_Final_Incremental_Model/Phase17_Categorical_Incremental_Value/Tables','03_paired_incremental_bootstrap.csv'),
 ('Phase 18 uncertainty','10_Final_Incremental_Model/Phase18_Final_Validation/Tables','01_model_metrics_and_uncertainty.csv'),
 ('Phase 18 paired comparison','10_Final_Incremental_Model/Phase18_Final_Validation/Tables','02_paired_model_C_vs_A.csv'),
 ('Phase 18 calibration','10_Final_Incremental_Model/Phase18_Final_Validation/Tables','03_calibration_bins.csv'),
 ('Phase 18 OPO performance','10_Final_Incremental_Model/Phase18_Final_Validation/Tables','04_per_OPO_model_C.csv'),
 ('Phase 11 clinical comparisons','09_Post_Procurement_Analysis/Brain_Death_Subgroup/Tables','03_observed_clinical_comparison.csv'),
 ('Phase 12 adjusted associations','09_Post_Procurement_Analysis/Brain_Death_Subgroup/Phase12_Oxygenation_Missingness/Tables','03_adjusted_associations.csv'),
 ('Phase 16 categories','09_Post_Procurement_Analysis/Brain_Death_Subgroup/Phase16_Reconstructed_Categories/Tables','03_adjusted_category_associations.csv'),
]
rows=[]
for label, relative, filename in sources:
    candidate = ROOT / relative / filename
    if not candidate.exists():
        matches = [p for p in ROOT.rglob(filename) if OUT not in p.parents]
        candidate = matches[0] if len(matches)==1 else None
    if candidate is not None and candidate.exists():
        dst=OUT/'Source_Copies'/f'{len(rows)+1:02d}_{filename}'
        shutil.copy2(candidate,dst)
        status='FOUND'
        source=str(candidate)
    else:
        status='MISSING_OR_AMBIGUOUS'
        source=str(ROOT / relative / filename)
    rows.append({'analysis':label,'status':status,'expected_or_found_source':source})

with (OUT/'Diagnostics'/'source_inventory.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

checklist = '''# Manuscript consolidation checklist\n\nStatus: SOURCE ASSEMBLY ONLY. Do not submit without resolving items below.\n\n1. Verify ORCHID dictionary: authorization, procurement, brain-death, ABO subtype and cause-of-death coding.\n2. Confirm dates and whether physiological measurements precede utilization decisions.\n3. Distinguish retrospective classification from prospective prediction and acceptance/suitability.\n4. Resolve source inventory missing/ambiguous files and reconcile duplicated filenames.\n5. Report denominators and missingness for all descriptive and regression analyses.\n6. Verify OPO-grouped CV fold composition and six-cluster uncertainty limitations.\n7. Separate prespecified tests from exploratory screening; avoid causal language.\n8. Cite original ORCHID source and relevant pediatric donor acceptability literature.\n9. Assemble Figure 1 flow, model performance, calibration and OPO heterogeneity figures.\n10. Check every manuscript number against a copied source table.\n'''
(OUT/'Diagnostics'/'manuscript_checklist.md').write_text(checklist,encoding='utf-8')
print('Consolidation folder:',OUT)
print('Found',sum(x['status']=='FOUND' for x in rows),'of',len(rows),'source tables')
print('Review Diagnostics/source_inventory.csv for any missing paths.')
