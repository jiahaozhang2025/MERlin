"""Move a dataset analyzed with the flat output layout into the grouped one.

In the grouped layout (see DataSet.__init__) each task folder sits under the
folder of its class's outputGroup -- Prepare, Optimize, Decode, Segment,
Export, Other -- and the top-level dataset files (codebooks, data
organization, positions, dataset.json ...) under Files.

    python -m merlin.util.regroup ANALYSIS_HOME/<dataset>          # show plan
    python -m merlin.util.regroup ANALYSIS_HOME/<dataset> --apply  # move

Every move is a rename inside the dataset folder, so nothing is copied, and
the list of moves is written to logs/ first so it can be reversed. The
snakemake and logs folders stay at the top level. Snakefiles written before
the move name the old paths: regenerate the workflow before running again.
"""

import argparse
import datetime
import importlib
import json
import os
import sys

from merlin.core import analysistask

KEEP_AT_TOP = ('snakemake', 'logs')


def _task_group(taskPath):
    with open(os.path.join(taskPath, 'tasks', 'task.json')) as f:
        parameters = json.load(f)
    try:
        module = importlib.import_module(parameters['module'])
        return getattr(module, parameters['class']).outputGroup
    except (ImportError, AttributeError) as e:
        print('  cannot load %s.%s for %s (%s); it goes to %s'
              % (parameters['module'], parameters['class'],
                 os.path.basename(taskPath), e,
                 analysistask.AnalysisTask.outputGroup))
        return analysistask.AnalysisTask.outputGroup


def _unfinished_fragments(taskPath):
    """Status files of fragments that started and wrote neither done nor
    error: still running, or killed without being marked."""
    names = set(os.listdir(os.path.join(taskPath, 'tasks')))
    return sorted(n for n in names if n.endswith('.start')
                  and n[:-6] + '.done' not in names
                  and n[:-6] + '.error' not in names)


def _escaping_relative_links(path, root):
    """Relative symlinks at path or one level inside it that point outside
    root; moving root to another depth would break them."""
    candidates = [path]
    if os.path.isdir(path) and not os.path.islink(path):
        candidates += [os.path.join(path, n) for n in os.listdir(path)]
    escaping = []
    for c in candidates:
        if os.path.islink(c) and not os.path.isabs(os.readlink(c)):
            target = os.path.normpath(
                os.path.join(os.path.dirname(c), os.readlink(c)))
            if os.path.commonpath([target, root]) != root or c == root:
                escaping.append(c)
    return escaping


def plan_moves(analysisPath):
    """Return (moves, problems): moves as (name, group) pairs, group 'Files'
    for top-level files; problems that should stop the move."""
    moves = []
    problems = []
    for name in sorted(os.listdir(analysisPath)):
        path = os.path.join(analysisPath, name)
        if name in KEEP_AT_TOP:
            continue
        if not os.path.isdir(path):
            moves.append((name, 'Files'))
        elif os.path.isfile(os.path.join(path, 'tasks', 'task.json')):
            moves.append((name, _task_group(path)))
            unfinished = _unfinished_fragments(path)
            if unfinished:
                problems.append('%s has %i unfinished fragment(s), e.g. %s'
                                % (name, len(unfinished), unfinished[0]))
        else:
            print('  %s/ is not a task folder; left in place' % name)
            continue
        for link in _escaping_relative_links(path, path):
            problems.append('relative symlink %s would break'
                            % os.path.relpath(link, analysisPath))
    # dataset.json marks the flat layout, so it moves last: an interrupted
    # move is then still read as flat and can be finished from the log
    moves.sort(key=lambda m: m[0] == 'dataset.json')
    return moves, problems


def apply_moves(analysisPath, moves):
    logPath = os.path.join(analysisPath, 'logs', 'regroup_%s.tsv'
                           % datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
    os.makedirs(os.path.dirname(logPath), exist_ok=True)
    with open(logPath, 'w') as f:
        for name, group in moves:
            f.write('%s\t%s\n' % (name, os.path.join(group, name)))
    print('move list written to %s' % logPath)

    # Through a staging folder, since a task can share its group's name
    # (a Segment task goes to Segment/Segment).
    staging = os.path.join(analysisPath, '.regroup_staging')
    os.makedirs(staging)
    for name, group in moves:
        os.rename(os.path.join(analysisPath, name),
                  os.path.join(staging, name))
    for name, group in moves:
        os.makedirs(os.path.join(analysisPath, group), exist_ok=True)
        os.rename(os.path.join(staging, name),
                  os.path.join(analysisPath, group, name))
    os.rmdir(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Move a flat MERlin analysis folder into the grouped '
                    'layout. Shows the plan unless --apply is given.')
    parser.add_argument('analysis_path',
                        help='the dataset folder, ANALYSIS_HOME/<dataset>')
    parser.add_argument('--apply', action='store_true',
                        help='perform the moves')
    parser.add_argument('--force', action='store_true',
                        help='move despite unfinished fragments or relative '
                             'symlinks')
    args = parser.parse_args(argv)

    analysisPath = os.path.abspath(args.analysis_path)
    if not os.path.isfile(os.path.join(analysisPath, 'dataset.json')):
        sys.exit('%s has no top-level dataset.json: not a flat dataset '
                 '(already grouped, or not a dataset folder)' % analysisPath)

    moves, problems = plan_moves(analysisPath)
    width = max(len(n) for n, _ in moves)
    for name, group in moves:
        print('  %-*s -> %s/%s' % (width, name, group, name))
    for p in problems:
        print('PROBLEM: ' + p)

    if not args.apply:
        print('dry run; add --apply to move')
        return
    if problems and not args.force:
        sys.exit('not moving: resolve the problems above or pass --force')
    apply_moves(analysisPath, moves)
    print('done; regenerate the snakemake workflow before running again')


if __name__ == '__main__':
    main()
