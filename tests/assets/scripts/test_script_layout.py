"""Verify script ownership, symlink identity and relocated execution paths."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
LIBRARY = ROOT / 'assets/tools/proteolysis_data_manager'
SHORTCUTS = {
    'assets/scripts/downloads/proteolysis_data_manager': 'assets/tools/proteolysis_data_manager/scripts/downloads',
    'assets/scripts/integration/proteolysis_data_manager': 'assets/tools/proteolysis_data_manager/scripts/integration',
    'assets/scripts/generation/blosum_matrix': 'assets/tools/blosum_matrix/scripts/generation',
    'assets/scripts/downloads/pep_compass': 'scripts/downloads',
}
PROTEOLYSIS_COMMANDS = [
    'downloads/download_all.py',
    'downloads/half_time_data/download_all.py',
    'downloads/half_time_data/download_peplife2.py',
    'downloads/half_time_data/download_pepmsnd.py',
    'downloads/half_time_data/download_plifepred.py',
    'downloads/half_time_data/download_dramp.py',
    'downloads/half_time_data/download_cavaco.py',
    'downloads/kinetics_data/import_brenda.py',
    'downloads/kinetics_data/download_all.py',
    'downloads/kinetics_data/download_sabio.py',
    'downloads/window_neighbourhood_data/download_all.py',
    'downloads/window_neighbourhood_data/download_merops.py',
    'downloads/window_neighbourhood_data/download_mmp_pics.py',
    'downloads/window_neighbourhood_data/download_dpp4_qpisa.py',
    'integration/half_time_data/normalize_datasets.py',
    'integration/kinetics_data/normalize_datasets.py',
    'integration/window_neighbourhood_data/merge_merops.py',
]


@pytest.mark.parametrize('shortcut,canonical', SHORTCUTS.items())
def test_shortcuts_are_relative_directory_symlinks(shortcut, canonical):
    link = ROOT / shortcut
    assert link.is_symlink()
    assert not Path(os.readlink(link)).is_absolute()
    assert link.samefile(ROOT / canonical)


def test_repository_script_tree_contains_no_executable_copies():
    for directory, _, files in os.walk(ROOT / 'assets/scripts', followlinks=False):
        assert not [name for name in files if Path(name).suffix in {'.py', '.sh'}], directory
    for name in ['half_time_data', 'kinetics_data', 'serum_proteolysis_calibration',
                 'download_proteolysis_raw.py', 'apex', 'hydraAMP', 'merops', 'pep-compass']:
        assert not (ROOT / 'assets/scripts/downloads' / name).exists()


@pytest.mark.parametrize('relative', PROTEOLYSIS_COMMANDS)
def test_library_and_shortcut_commands_work_from_another_directory(relative, tmp_path):
    operation, rest = relative.split('/', 1)
    canonical = LIBRARY / 'scripts' / relative
    shortcut = ROOT / 'assets/scripts' / operation / 'proteolysis_data_manager' / rest
    assert canonical.samefile(shortcut)
    outputs = []
    for path in [canonical, shortcut]:
        result = subprocess.run([sys.executable, str(path), '--help'], cwd=tmp_path,
                                text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert 'usage:' in result.stdout
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_download_launchers_delegate_only_to_acquisition_cli():
    allowed = {'download_raw', 'download_half_life', 'import_brenda',
               'download_window_neighbourhood', 'download_kinetics'}
    for path in (LIBRARY / 'scripts/downloads').rglob('*.py'):
        tree = ast.parse(path.read_text())
        delegates = [alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                     and node.module == 'proteolysis_data_manager.cli' for alias in node.names]
        assert len(delegates) == 1 and delegates[0] in allowed, path
        assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body), path


def test_acquisition_orchestrators_are_explicitly_named_as_all_modules():
    acquisition = LIBRARY / 'src/proteolysis_data_manager/acquisition'
    assert (acquisition / 'download_all.py').is_file()
    assert (acquisition / 'half_life/download_all.py').is_file()
    assert (acquisition / 'kinetics/download_all.py').is_file()
    assert (acquisition / 'window_neighbourhood/download_all.py').is_file()
    assert not (acquisition / 'download.py').exists()
    assert not (acquisition / 'half_life/download.py').exists()
    assert not (acquisition / 'kinetics/download.py').exists()
    assert (acquisition / 'helpers/raw.py').is_file()
    assert not (acquisition / '_raw.py').exists()


def test_download_launchers_declare_repository_output_roots():
    library_scripts = LIBRARY / 'scripts/downloads'
    all_script = library_scripts / 'download_all.py'
    spec = importlib.util.spec_from_file_location('proteolysis_download_all_script', all_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_HALF_LIFE_ROOT == ROOT / 'data/proteolysis/half_time_datasets'
    assert module.DEFAULT_KINETICS_ROOT == ROOT / 'data/proteolysis/kinetics_datasets'
    assert module.DEFAULT_WINDOW_ROOT == ROOT / 'data/proteolysis/window_neighbourhood_datasets'

    for relative, dataset in [
        ('window_neighbourhood_data/download_all.py', 'window_neighbourhood_datasets'),
        ('window_neighbourhood_data/download_merops.py', 'window_neighbourhood_datasets'),
        ('window_neighbourhood_data/download_mmp_pics.py', 'window_neighbourhood_datasets'),
        ('window_neighbourhood_data/download_dpp4_qpisa.py', 'window_neighbourhood_datasets'),
        ('kinetics_data/download_all.py', 'kinetics_datasets'),
        ('kinetics_data/download_sabio.py', 'kinetics_datasets'),
    ]:
        spec = importlib.util.spec_from_file_location('source_script', library_scripts / relative)
        source_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source_module)
        assert source_module.DEFAULT_DATA_ROOT == ROOT / 'data/proteolysis' / dataset

    for source in ['download_all', 'download_peplife2', 'download_pepmsnd',
                   'download_plifepred', 'download_dramp', 'download_cavaco']:
        path = library_scripts / f'half_time_data/{source}.py'
        spec = importlib.util.spec_from_file_location(f'proteolysis_{source}_script', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.DEFAULT_DATA_ROOT == ROOT / 'data/proteolysis/half_time_datasets'

    brenda = library_scripts / 'kinetics_data/import_brenda.py'
    spec = importlib.util.spec_from_file_location('proteolysis_import_brenda_script', brenda)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_DATA_ROOT == ROOT / 'data/proteolysis/kinetics_datasets'


@pytest.mark.parametrize('script', ['export_blosum62.py', 'generate_ampblosum62.py', 'generate_all.py'])
def test_blosum_shortcuts_load_the_library_from_another_directory(script, tmp_path):
    path = ROOT / 'assets/scripts/generation/blosum_matrix' / script
    result = subprocess.run([sys.executable, str(path), '--help'], cwd=tmp_path,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'Usage:' in result.stdout


def test_blosum_default_paths_still_point_into_the_repository():
    path = ROOT / 'assets/scripts/generation/blosum_matrix/_paths.py'
    spec = importlib.util.spec_from_file_location('relocated_blosum_paths', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PACKAGE_ROOT == ROOT / 'assets/tools/blosum_matrix'
    assert module.REPO_ROOT == ROOT
    assert module.DEFAULT_OUT_DIR == ROOT / 'data/peptides_data/blosum_matrix'


def test_shared_data_downloader_preserves_repository_default(tmp_path):
    path = ROOT / 'assets/scripts/downloads/pep_compass/data/download_data_folder.py'
    spec = importlib.util.spec_from_file_location('relocated_shared_data_download', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._repository_root() == ROOT
    result = subprocess.run([sys.executable, str(path), '--help'], cwd=tmp_path,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('relative,environment,marker', [
    ('apex/download_apex_models_default.sh', 'APEX_MODELS_DIR', 'APEX_fixture'),
    ('apex/download_apex_models_full.sh', 'APEX_MODELS_DIR', 'trained_fixture'),
    ('hydramp/download_hydramp_model.sh', 'HYDRAMP_MODELS_DIR', 'fixture.pickle'),
    ('merops/download_merops_matrices.sh', 'MEROPS_DATA_DIR', 'index.json'),
])
def test_relocated_model_downloaders_recognize_existing_artifacts(relative, environment, marker, tmp_path):
    target = tmp_path / 'existing'
    target.mkdir()
    (target / marker).write_text('fixture')
    script = ROOT / 'assets/scripts/downloads/pep_compass' / relative
    result = subprocess.run(['bash', str(script)], cwd=tmp_path,
                            env=os.environ | {environment: str(target)},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'Skipping download' in result.stdout + result.stderr
    assert str(target) in result.stdout + result.stderr
