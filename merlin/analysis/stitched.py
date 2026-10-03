"""Opt-in stitched-volume tasks; existing per-FOV tasks are unchanged.

Each task participates in MERlin's usual dependency/fragment/status machinery.
Numerical stages run in subprocesses so legacy scientific adapters cannot alter
the importing MERlin process. A frozen source copy belongs to each new run.
"""
from pathlib import Path
from merlin.core import analysistask
from merlin.util.stitched import workflow


class _Task:
    def get_dependencies(self):
        return list(self.parameters.get('dependencies', []))

    def get_estimated_memory(self):
        return self.parameters.get('memory_mb', 16384)

    def get_estimated_time(self):
        return self.parameters.get('minutes', 120)

    def _initial(self):
        return self.dataSet.load_analysis_task(self.parameters['initialize_task'])

    def _root(self):
        return Path(self.dataSet.get_analysis_subdirectory(
            self.parameters['initialize_task'], 'stitched_run'))

    def _execute(self, fragment=None):
        workflow.execute(self._root(), self.parameters['stage'], fragment,
                         self.parameters.get('workers', 52))


class StitchedInitialize(_Task, analysistask.AnalysisTask):
    """Freeze inputs and operators, validate acquisition schedules and scope."""
    def _run_analysis(self):
        root=Path(self.dataSet.get_analysis_subdirectory(self, 'stitched_run'))
        workflow.initialize(root, workflow.configuration(self.dataSet, self.parameters))

    def get_dependencies(self):
        return []


class StitchedStage(_Task, analysistask.AnalysisTask):
    """A serial stitched stage, including review, filtering and export."""
    def _run_analysis(self):
        self._execute()


class StitchedFragments(_Task, analysistask.ParallelAnalysisTask):
    """Native MERlin fragments for extraction, registration, decode and joins."""
    def fragment_count(self):
        config=workflow.configuration(self.dataSet, self._initial().parameters)
        stage=self.parameters['stage']
        if stage in ('extract','qa'):
            return len(config['rounds'])*52
        if stage in ('within','pool','cross'):
            return len(config['rounds'])
        if stage=='calibration_images':
            return 62
        if stage=='decode':
            return int(self.parameters.get('workers',52))
        if stage=='deduplicate':
            return config['codeword_count']
        if stage in ('partition_geometry','partition_join'):
            return 52
        raise ValueError('Unknown parallel stitched stage: '+stage)

    def _run_analysis(self, fragmentIndex):
        self._execute(int(fragmentIndex))
