"""Run in a new notebook with %run, or as a Python script on the data server."""
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import json
import uuid

D = Path('/home/seowon1211/data1')
BASE = D / 'conditioned_expression'
# To resume, replace None with the printed ROOT path. Keep its code snapshot intact.
RESUME_ROOT = None
ROOT = Path(RESUME_ROOT) if RESUME_ROOT else BASE / (
    'complete_pca_full_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
CODE = ROOT / 'code'
PREPARED = BASE / 'prepared_delta'
STRUCTURE = D / 'struct_feats/struct_morgan2048.csv'
SMILES = D / 'compound_pubchem_smiles.csv'
for path in [PREPARED/'preparation.json', PREPARED/'pairs.csv.gz',
             PREPARED/'vitro_expression.npy', PREPARED/'vivo_expression.npy', STRUCTURE, SMILES]:
    if not path.is_file():
        raise FileNotFoundError(f'{path}\nSet PREPARED, STRUCTURE or SMILES to the existing input file before running.')
if RESUME_ROOT:
    if not (CODE/'scripts/compare_input_pca_full.py').is_file():
        raise FileNotFoundError(CODE)
else:
    ROOT.mkdir(parents=True, exist_ok=False)
    subprocess.run(['git','clone','https://github.com/kiseowon/diffusion1st.git',str(CODE)],check=True)
    # Freeze the checked-out commit. No pull on resume.
revision = subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
cmd = [sys.executable,'-u',str(CODE/'scripts/compare_input_pca_full.py'),
       '--prepared',str(PREPARED),'--structure',str(STRUCTURE),'--smiles',str(SMILES),
       '--out',str(ROOT/'results'),'--arms','basic','morgan','descriptor',
       '--families','mlp','cgan','ddpm','--folds','5','--split-seed','42',
       '--model-seeds','42','43','44','--epochs','100','--threads','4',
       '--budget-minutes','10080','--unit-budget-minutes','10080']
print('ROOT:', ROOT, '\nCode commit:', revision, flush=True)
subprocess.run(cmd,check=True)  # plan only
manifest = ROOT/'launcher.json'
record = dict(commit=revision,command=cmd)
if manifest.exists():
    if json.loads(manifest.read_text()) != record:
        raise ValueError('Resume launcher/code configuration differs')
else:
    manifest.write_text(json.dumps(record,indent=2),encoding='utf-8')
with (ROOT/'training.log').open('a',encoding='utf-8') as log:
    with subprocess.Popen(cmd+['--execute'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,bufsize=1) as process:
        for line in process.stdout:
            print(line,end='',flush=True)
            log.write(line)
            log.flush()
        status=process.wait()
if status:
    raise RuntimeError(f'Exit {status}; inspect {ROOT / "training.log"}')
status_path=ROOT/'results/comparison_status.json'
state=json.loads(status_path.read_text())
if not state['all_complete']:
    print('Partial run. Set RESUME_ROOT to',repr(str(ROOT)),'and rerun the same snapshot.')
else:
    import pandas as pd
    print(pd.read_csv(ROOT/'results/comparison_summary.csv').to_string(index=False))
    print('Classification:',ROOT/'results/combined_classification_summary.csv')
    print('All metrics:',ROOT/'results/pooled_metrics.csv')
