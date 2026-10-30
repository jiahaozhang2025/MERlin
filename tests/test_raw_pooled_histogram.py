import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd

from merlin.analysis.rawcalibration import PooledRawHistogram


class PooledHistogramTests(unittest.TestCase):
    def task(self, missing=False):
        tables = [pd.DataFrame(columns=['barcode_id','mean_intensity','min_distance','area']),
            pd.DataFrame(dict(barcode_id=[0,1],mean_intensity=[10.,100.],min_distance=[.2,.1],area=[3,2])),
            pd.DataFrame(dict(barcode_id=[0],mean_intensity=[1000.],min_distance=[.3],area=[5]))]
        outputs = {}
        database = SimpleNamespace(get_barcodes=lambda fov,columnList: tables[fov][columnList])
        book = SimpleNamespace(get_blank_indexes=lambda:[1],get_coding_indexes=lambda:[0])
        decode = SimpleNamespace(parameters={'distance_threshold':.65},
            is_complete=lambda f:not (missing and f==2),get_barcode_database=lambda:database,get_codebook=lambda:book)
        task = object.__new__(PooledRawHistogram)
        task.parameters = dict(decode_task='Decode', fovs=[0,1,2], intensity_bins=20,
            distance_bins=20, area_bins=20, intensity_transform='log10')
        task.dataSet = SimpleNamespace(load_analysis_task=lambda n:decode,get_fovs=lambda:[0,1,2],
            save_numpy_analysis_result=lambda value,name,*a:outputs.update({name:value}),
            save_json_analysis_result=lambda value,name,*a:outputs.update({name:value}))
        return task, outputs

    def test_empty_field_does_not_destroy_global_intensity_range(self):
        task, output = self.task();task._run_analysis()
        self.assertEqual(output['intensity_bins'][-1],6.)
        self.assertEqual(output['coding_counts'].sum(),2)
        self.assertEqual(output['blank_counts'].sum(),1)
        self.assertEqual(output['binning_provenance']['empty_fovs'],[0])
        self.assertTrue(output['complete_fragments'].all())
        empty = pd.DataFrame(columns=['barcode_id','mean_intensity','min_distance','area'])
        self.assertTrue(task.extract_barcodes_with_threshold(.1, empty).empty)

    def test_incomplete_field_cannot_produce_partial_pooled_filter(self):
        task, _ = self.task(missing=True)
        with self.assertRaisesRegex(RuntimeError,'all selected decode fields'):
            task._run_analysis()


if __name__ == '__main__':
    unittest.main()
