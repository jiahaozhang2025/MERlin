import csv,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from merlin.util.stitched import workflow,configure,registration_review
from merlin.analysis import stitched

OPERATORS=Path(workflow.__file__).parent/'operators'
sys.path.insert(0,str(OPERATORS))
from tile_layout import owner_for_global_xy,MPP
from full_decode_operators import own_components,DecodeHaloTooSmall
from full_filter_operators import deduplicate_identity,fit_streaming_threshold
from merlin.util import barcodefilters
from merlin.analysis import filterbarcodes
from merlin.util.stitched.cell_centers import map_fragments


class StitchedTests(unittest.TestCase):
    def metadata(self,folder,index=1):
        folder.mkdir(exist_ok=True)
        (folder/f'codebook_{index}_test.csv').write_text('name,id,A,B,C,D\nGene1,a,1,1,0,0\nGene2,b,0,0,1,1\nBlank1,c,1,0,1,0\n')
        (folder/'dataorganization.csv').write_text('readoutName,imagingRound\nA,10\nB,11\nC,12\nD,12\n')
        return SimpleNamespace(analysisPath=str(folder),rawDataPath=str(folder/'raw'))

    def test_requested_codebook_and_dependency_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds=self.metadata(Path(tmp));config=dict(profile='0718_40x_150z',codebook_index=1,metadata_dir=tmp)
            c=workflow.configuration(ds,{'configuration':config})
            self.assertEqual(c['rounds'],[0,10,11,12]);self.assertEqual(c['codeword_count'],3)
            pipeline=configure.make_pipeline(config,partition=True,report=True)
            tasks={}
            ds.load_analysis_task=lambda name:tasks[name]
            for entry in pipeline['analysis_tasks']:
                cls=getattr(stitched,entry['task']);t=cls(ds,entry['parameters'],entry['analysis_name']);tasks[t.analysisName]=t
                self.assertTrue(set(t.get_dependencies()).issubset(tasks))
            self.assertEqual(tasks['StitchedExtract'].fragment_count(),208)
            self.assertEqual(tasks['StitchedDeduplicate'].fragment_count(),3)
            self.assertEqual(tasks['StitchedCalibrationImages'].fragment_count(),62)
            self.assertIn('StitchedReview',tasks['StitchedCalibrationPlan'].get_dependencies())
            self.assertIn('StitchedPartitionVerify',tasks['StitchedCellCenters'].get_dependencies())
            self.assertIn('StitchedCellCenters',tasks['StitchedAnalysis'].get_dependencies())
            with self.assertRaises(ValueError):workflow.configuration(ds,{'configuration':dict(config,codebook_index=0)})

    def test_submicron_boundary_unique_owner(self):
        layout={'origin_pixel_xy':[0,0],'shape_tiles_yx':[1,2]}
        x=np.array([[2048*MPP-1e-5,4],[2048*MPP,4],[2048*MPP+1e-5,4]])
        np.testing.assert_array_equal(owner_for_global_xy(x,layout),[0,1,1])

    def test_truncated_component_is_not_silently_counted(self):
        labels=np.zeros((12,12),int);labels[1:6,4:7]=1
        frame=pd.DataFrame(dict(unique_id=[1],global_x=[.5],global_y=[.5],x=[4.],y=[4.]))
        with self.assertRaises(DecodeHaloTooSmall):
            own_components(frame,labels,{'shape_yx':[8,8],'tile_id':0},{'origin_pixel_xy':[0,0],'shape_tiles_yx':[1,1]},2,0)

    def test_cross_tile_cross_z_dedup_is_global(self):
        frame=pd.DataFrame(dict(barcode_uid=['a','b','c'],barcode_id=[0,0,0],global_x=np.array([2047.8,2048.2,3000])*MPP,
            global_y=[4.,4.,4.],z=[0,1,1],mean_intensity=[10.,20.,15.],output_tile=[0,1,1]))
        kept,removed=deduplicate_identity(frame,[1.,2.],barcodefilters.remove_zplane_duplicates_single_barcodeid)
        self.assertEqual(set(kept.barcode_uid),{'b','c'});self.assertEqual(set(removed.barcode_uid),{'a'})
        pd.testing.assert_frame_equal(kept,frame.loc[[1,2]])

    def test_neighbor_center_recovers_own_outside_support(self):
        offsets={'own':{'0':{'t':[2000,0],'how':'test'},'1':{'t':[0,0],'how':'test'}},'pairs':{'1_0':{'t':[0,0],'how':'test'}}}
        setup={'offsets':offsets,'crops':{'0':[0,0],'1':[0,0]}}
        cells=pd.DataFrame(dict(stack=[0],label=[1],cell_uid=[7],cx_ds=[100],cy_ds=[100],cz_ds=[40],nvox=[10]))
        out=map_fragments(cells,{'models':{'0':{},'1':{}}},setup,lambda p,f,m:p+np.array([f*10,0,0]))
        self.assertEqual(out.reference_fov.iloc[0],1);self.assertTrue(out.used_neighbor.iloc[0]);self.assertEqual(out.cell_uid.iloc[0],7)
        self.assertAlmostEqual(out.global_x.iloc[0],200*MPP+10)

    def test_qa_does_not_treat_sparse_or_ambiguous_as_success(self):
        good=dict(status='IDENTIFIABLE',xy_residual_px=.5,z_residual_um=.2)
        self.assertFalse(registration_review.metrics([good]*7)['qualified_gates_pass'])
        self.assertTrue(registration_review.metrics([good]*8)['qualified_gates_pass'])
        self.assertFalse(registration_review.metrics([dict(status='AMBIGUOUS_IMAGE_MATCH')]*100)['qualified_gates_pass'])

    def test_optimizer_preserves_absolute_intensity_scale(self):
        from merlin.analysis.optimize import OptimizeIteration
        from fresh_pipeline.fresh_optimize import canonical_aggregate
        scales,bg,_=canonical_aggregate(OptimizeIteration,np.array([50.,100.]),np.zeros(2),
            [np.array([1.,1.]),np.array([1.,1.])],[np.zeros(2),np.zeros(2)])
        np.testing.assert_array_equal(scales,[50.,100.])
        np.testing.assert_array_equal(bg,[0.,0.])

    def test_cluster_resources_cover_optimizer_workers(self):
        p=configure.make_pipeline(dict(num_threads=2,optimizer_workers=4))
        r=configure.cluster_resources(p)
        self.assertEqual(r['StitchedOptimize']['cpus'],8)
        self.assertEqual(r['StitchedDecode']['cpus'],2)
        self.assertEqual(r['StitchedDecodeDone']['cpus'],1)

    def test_current_canonical_streaming_adaptive_uses_real_blank_counts(self):
        book=SimpleNamespace(get_blank_indexes=lambda:[2],get_coding_indexes=lambda:[0,1],get_barcode_count=lambda:3)
        frame=pd.DataFrame([dict(barcode_id=i%2,mean_intensity=10.,min_distance=.1,area=7) for i in range(1000)]+
            [dict(barcode_id=2,mean_intensity=.5,min_distance=.6,area=30) for _ in range(50)])
        task,result=fit_streaming_threshold(lambda:iter([frame.iloc[:517],frame.iloc[517:]]),book,filterbarcodes)
        self.assertEqual(result['raw_barcodes'],1050);self.assertEqual(result['raw_blank_barcodes'],50)
        self.assertTrue(result['histogram_count_conservation_passed']);self.assertTrue(np.isfinite(result['blank_fraction_threshold']))

if __name__=='__main__':unittest.main()
