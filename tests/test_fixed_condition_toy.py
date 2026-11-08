import unittest,contextlib,io,tempfile,sys,copy,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import fixed_condition_toy as toy
import repeated_latent_validation as core

class ToyTests(unittest.TestCase):
    def test_plan_no_io(self):
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(toy.main(['--prepared','/missing','--structure','/missing','--smiles','/missing','--out','/missing']),27)

    def make(self,root):
        import pandas as pd
        from test_repeated_latent_validation import RepeatedLatentTests
        a=RepeatedLatentTests().synthetic(root);smiles=root/'smiles.csv'
        pd.DataFrame({'compound':[f'compound_{i:02}' for i in range(20)],'smiles':['C'*(i+2) for i in range(20)]}).to_csv(smiles,index=False)
        _,data,ids,meta,Y,_,_=core.load_data(a)
        source=pd.read_csv(a.prepared/'vitro_conditions.csv');source=source[source.condition_id.isin(data.source_ids)].reset_index(drop=True)
        raw,audit,_=toy.chemistry.calculate_descriptors(smiles)
        return a,smiles,data,meta,Y,source,raw,audit

    def test_cohort_and_training_only_transforms(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd,numpy as np,torch
        with tempfile.TemporaryDirectory() as d:
            a,smiles,data,meta,Y,source,raw,audit=self.make(Path(d))
            table,sids,target,eligibility=toy.cohort(data,meta,Y,source,raw,audit)
            self.assertEqual(len(table),20)
            np.testing.assert_array_equal(table.source_time_h,24)
            fit=np.arange(12)
            packs=toy.prepare(data,table,sids,target,raw,fit,4)
            changed=target.copy();changed[12:]+=1000
            changed_data=copy.deepcopy(data);changed_data.source[sids[12:]]+=1000
            changed_raw=raw.copy();changed_raw.loc[table.compound.iloc[12:]]=changed_raw.loc[table.compound.iloc[12:]]+1000
            other=toy.prepare(changed_data,table,sids,changed,changed_raw,fit,4)
            for arm in packs:
                torch.testing.assert_close(packs[arm]['X'][fit],other[arm]['X'][fit])
                torch.testing.assert_close(packs[arm]['Z'],other[arm]['Z'])
                self.assertEqual(torch.count_nonzero(packs[arm]['Z'][12:]).item(),0)
            self.assertEqual(len({p['X'].shape[1] for p in packs.values()}),1)
            self.assertEqual(torch.count_nonzero(packs['basic']['X'][:,4:]).item(),0)
            # Availability/uniqueness exclusions are shared across all arms.
            data.structure[0,-1]=0
            raw.loc['compound_01']=np.nan
            duplicate=source[source.compound.eq('compound_02')&source.level.eq('High')]
            modified=pd.concat([source,duplicate],ignore_index=True)
            table,_,_,eligibility=toy.cohort(data,meta,Y,modified,raw,audit)
            self.assertEqual(len(table),17)
            self.assertFalse(eligibility.set_index('compound').loc[['compound_00','compound_01','compound_02'],'included'].any())

    def test_nine_jobs_complete_resume_and_contract(self):
        try:import rdkit
        except ImportError:self.skipTest('RDKit missing')
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a,smiles,*_=self.make(root)
            argv=['--prepared',str(a.prepared),'--structure',str(a.structure),'--smiles',str(smiles),'--out',str(root/'toy'),'--input-components','4','--epochs','1','--model-seeds','42','--threads','1','--execute']
            with contextlib.redirect_stdout(io.StringIO()):toy.main(argv)
            result=pd.read_csv(root/'toy/by_seed.csv');self.assertEqual(len(result),11)
            self.assertEqual(result.n_probes.unique().tolist(),[6])
            self.assertEqual(result.n_targets.unique().tolist(),[4])
            status=json.loads((root/'toy/complete.json').read_text());self.assertTrue(status['all_complete']);self.assertFalse(status['all_folds_evaluated'])
            for kind in ['mlp','cgan','ddpm']:
                counts=[json.loads(p.read_text())['model_parameters'] for p in (root/'toy').glob(f'*/seed_42/{kind}/attempt_*/model_dimensions.json')]
                self.assertEqual(len(counts),3);self.assertEqual(len(set(counts)),1)
            with patch.object(core,'fit_job',side_effect=AssertionError('No training on resume')):
                with contextlib.redirect_stdout(io.StringIO()):toy.main(argv)
            changed=argv.copy();changed[changed.index('--input-components')+1]='3'
            with self.assertRaises(ValueError):
                with contextlib.redirect_stdout(io.StringIO()):toy.main(changed)
