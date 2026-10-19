"""Behavioral checks for scientific label semantics and acquisition provenance."""
from pathlib import Path
import importlib.util
import io
import json
import sys

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets/tools/proteolysis_data_manager/src'))
sys.path.insert(0, str(REPO / 'assets/scripts/downloads/half_time_data'))
from proteolysis_data_manager.integration.half_life.normalization import measurement, medium, peplife_row, chemical_signature, features, known
from proteolysis_data_manager.analysis.half_life.overview import split_manifest
import half_time_sources as acquisition


@pytest.mark.parametrize('raw,unit,relation,field,seconds', [
    ('2','hours','eq','half_life_seconds',7200),
    ('<30','minutes','lt','upper_bound_seconds',1800),
    ('≥24','hours','ge','lower_bound_seconds',86400),
    ('~1.5','hours','approx','approximate_half_life_seconds',5400),
    ('6 ·2','hours','eq','half_life_seconds',22320),
])
def test_units_and_censoring_are_preserved(raw, unit, relation, field, seconds):
    r = measurement(raw, unit)
    assert r['relation'] == relation
    assert r[field] == seconds
    if relation in {'lt','ge'}:
        assert r['half_life_seconds'] is None


def test_interval_is_not_replaced_by_its_midpoint():
    r = measurement('2–4', 'hours')
    assert (r['relation'],r['lower_bound_seconds'],r['upper_bound_seconds']) == ('interval',7200,14400)
    assert r['half_life_seconds'] is None


@pytest.mark.parametrize('raw', ['0','-2','nan','Stability increased'])
def test_invalid_or_unresolved_values_do_not_become_exact_labels(raw):
    assert measurement(raw, 'hours')['half_life_seconds'] is None


def test_percentage_at_time_does_not_become_half_life():
    r = measurement('More than 70% of peptides remain intact after 6.5 h.', prose=True)
    assert r['measurement_kind'] == 'remaining_fraction'
    assert r['remaining_fraction'] == .7
    assert r['fraction_relation'] == 'gt'
    assert r['observation_time_seconds'] == 23400
    assert r['half_life_seconds'] is None


@pytest.mark.parametrize('raw', [
    'the antibacterial activity was maintained up to 24h incubation in serum',
    'The antimicrobial activity was lost after 3h',
    '58% remaining after 24 hours, 56% remaining after 48 hours',
])
def test_activity_or_fraction_times_are_not_numeric_half_lives(raw):
    assert measurement(raw, prose=True)['half_life_seconds'] is None


def test_explicit_half_life_has_priority_over_complete_degradation_time():
    r = measurement('Degraded quickly with half-lives of less than 0.5 h and are completely gone after 2 h.', prose=True)
    assert r['relation'] == 'lt'
    assert r['upper_bound_seconds'] == 1800


def test_only_explicit_human_serum_is_identified():
    assert medium('Serum (human)') == ('human','serum')
    assert medium('Mouse serum') == ('mouse','serum')
    assert medium('Human plasma') == ('human','plasma')
    assert medium('Human serum and plasma') == ('human','mixed')
    assert medium('serum') == ('unknown','serum')


def test_chemistry_is_not_erased_and_sequences_are_not_clipped():
    raw = dict(id='1', seq='A'*70, half_life='<30',units_half='minutes',chiral='L',lin_cyc='Linear',nter='Free',cter='Free',chem_mod='None')
    free = peplife_row(raw)
    amidated = peplife_row(raw | {'cter':'Amidation'})
    assert len(free['sequence']) == 70
    assert chemical_signature(free) != chemical_signature(amidated)
    assert free['half_life_seconds'] is None
    assert 'reference_pi' in features(free['sequence'])
    assert not known('N.A.')
    assert not known('None')


def test_api_completeness_is_checked_against_fasta(tmp_path):
    folder = tmp_path/'raw/peplife2';folder.mkdir(parents=True)
    for filename, ids in [('natural.json',['1']),('modified.json',['2'])]:
        (folder/filename).write_text(json.dumps({'status':200,'count':len(ids),'data':[{'id':v} for v in ids]}))
    (folder/'complete.fasta').write_text('>1\nAAAA\n>2\nCCCC\n')
    assert len(acquisition.peplife_records(tmp_path)) == 2
    (folder/'complete.fasta').write_text('>1\nAAAA\n>2\nCCCC\n>3\nDDDD\n')
    with pytest.raises(ValueError, match='differs'):
        acquisition.peplife_records(tmp_path)


def test_api_rejects_embedded_errors_and_mismatched_counts(tmp_path):
    p=tmp_path/'response.json'
    for payload in [{'status':404,'count':0,'data':[]},{'status':200,'count':3,'data':[]}]:
        p.write_text(json.dumps(payload))
        with pytest.raises(ValueError):acquisition.validate_file(p)


def test_retained_file_preserves_original_receipt_and_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setitem(acquisition.SOURCES,'test',{'raw.json':'https://example.test/data'})
    calls=[]
    def retrieve(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(b'{"status":200,"count":0,"data":[]}')
    monkeypatch.setattr(acquisition,'urlopen',retrieve)
    acquisition.acquire('test',tmp_path)
    manifest=tmp_path/'manifests/download_test.json'
    original=manifest.read_bytes()
    acquisition.acquire('test',tmp_path)
    assert calls == ['https://example.test/data']
    assert manifest.read_bytes() == original
    legacy=tmp_path/'raw/test/download_manifest.json'
    manifest.replace(legacy)
    acquisition.acquire('test',tmp_path)
    assert manifest.read_bytes()==original and not legacy.exists()
    assert calls==['https://example.test/data']
    (tmp_path/'raw/test/raw.json').write_text('{"status":200,"count":0,"data":[],"extra":1}')
    with pytest.raises(ValueError,match='changed'):acquisition.acquire('test',tmp_path)


def test_failed_download_does_not_replace_previous_raw_file(tmp_path, monkeypatch):
    monkeypatch.setitem(acquisition.SOURCES,'test',{'raw.json':'https://example.test/data'})
    monkeypatch.setattr(acquisition,'urlopen',lambda *a,**k: io.BytesIO(b'{"status":200,"count":0,"data":[]}'))
    acquisition.acquire('test',tmp_path)
    original=(tmp_path/'raw/test/raw.json').read_bytes()
    monkeypatch.setattr(acquisition,'urlopen',lambda *a,**k: io.BytesIO(b'{"status":404,"count":0,"data":[]}'))
    with pytest.raises(ValueError):acquisition.acquire('test',tmp_path,force=True)
    assert (tmp_path/'raw/test/raw.json').read_bytes() == original


def test_evaluation_groups_connect_shared_sequences_and_publications():
    frame=pd.DataFrame([
        dict(observation_id='a',source='p',sequence='AAA',publication_id='10',baseline_eligible=True),
        dict(observation_id='b',source='q',sequence='AAA',publication_id='',baseline_eligible=True),
        dict(observation_id='c',source='p',sequence='CCC',publication_id='10',baseline_eligible=True),
        dict(observation_id='d',source='q',sequence='DDD',publication_id='',baseline_eligible=True),
    ])
    frame['is_human_serum']=True;frame['is_redundant_representation']=False
    result=split_manifest(frame).set_index('observation_id')
    assert result.loc['a','evaluation_group'] == result.loc['b','evaluation_group'] == result.loc['c','evaluation_group']
    assert result.loc['d','evaluation_group'] != result.loc['a','evaluation_group']


def test_serum_routing_uses_canonical_downloader():
    scripts=REPO/'assets/scripts/downloads/serum_proteolysis_calibration'
    sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('half_life_serum_router',scripts/'download_serum_sources.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for source,expected in [('peplife2_serum','peplife2'),('peplife2_plasma','peplife2'),('dramp_stability','dramp')]:
        command=module.source_command(source,Path('/tmp/manifest.tsv'),False)
        assert Path(command[1]).parent.name == 'half_time_data'
        assert command[-2:] == ['--source',expected]
    assert Path(module.source_command('hpa_blood_ms',Path('/tmp/manifest.tsv'),False)[1]).parent == scripts


def test_condition_selection_separates_unknown_and_known_mismatch():
    from proteolysis_data_manager.protocols.half_life.conditions import condition_status
    frame = pd.DataFrame({'ph': [7.4, None, None, 7.6, 7.35, 7.45],
                          'temperature_c': [37, 37, 40, None, 36, 38]})
    bounds = {'ph': [7.35, 7.45], 'temperature_c': [36, 38]}
    assert condition_status(frame, bounds).tolist() == ['accepted', 'unknown', 'outside', 'outside', 'accepted', 'accepted']
    assert condition_status(frame, {'ph': None}).eq('accepted').all()
    with pytest.raises(ValueError):
        condition_status(frame, {'ph': [8, 7]})


def test_condition_parsing_keeps_doses_and_mass_units_unresolved():
    from proteolysis_data_manager.protocols.half_life.conditions import condition_frame
    frame = pd.DataFrame({'peptide_concentration_raw': ['25 nM', '0.5 mM after 1:1 mixing', '1 mg/ml', '5 µM', '74 pmol/kg'],
                          'test_sample_raw': ['25% Human serum', 'Human serum', 'Human serum', '50% human serum', 'Human serum'],
                          'matrix_fraction': [None, .5, None, None, None], 'ph': [None]*5, 'temperature_c': [None]*5})
    result = condition_frame(frame)
    assert result.peptide_concentration_um.iloc[0] == .025
    assert result.peptide_concentration_um.iloc[1] == 500
    assert result.peptide_concentration_um.iloc[2:].isna().tolist() == [True, False, True]
    assert result.matrix_fraction.iloc[0] == .25
    assert result.matrix_fraction.iloc[1] == .5
    assert result.ph.isna().all()
    assert result.temperature_c.isna().all()


def test_reported_plus_minus_preserves_point_and_unknown_error_without_ci():
    r=measurement('1.57 ± 0.14','minutes')
    assert r['half_life_seconds'] == pytest.approx(94.2)
    assert r['reported_error_seconds'] == pytest.approx(8.4)
    assert r['uncertainty_kind'] == 'unspecified_plus_minus'
    original=peplife_row(dict(id='x',seq='AAA',half_life='1.57 ± 0.14',units_half='minutes'))
    assert original['ci_lower_seconds'] is None
    assert original['ci_upper_seconds'] is None
    assert original['half_life_sd_seconds'] is None
    assert original['half_life_mean_seconds'] is None


def test_dramp_splits_explicit_measurements_without_mixing_conditions():
    from proteolysis_data_manager.integration.half_life.normalization import dramp_rows
    raw={'DRAMP_ID':'DRAMP29881','Sequence':'AAA','Type':'Mouse serum',
         'half_life':'The half-live is 68 ±15 min in 25% aqueous mouse serum, 25±4 min in full mouse serum.'}
    rows=dramp_rows(raw)
    assert len(rows)==2
    assert [r['matrix_fraction'] for r in rows] == [.25,1.]
    assert [r['half_life_seconds'] for r in rows] == [4080,1500]
    assert [r['reported_error_seconds'] for r in rows] == [900,240]
    assert len(set(r['observation_id'] for r in rows))==2
    assert all(r['source_record_id']=='DRAMP29881' for r in rows)
    assert all(r['half_life_raw']==raw['half_life'] for r in rows)
    curve=raw | {'half_life':'Half-life is less than 0.5 h; complete degradation after 2 h in mouse serum.'}
    assert len(dramp_rows(curve)) == 1


def test_fraction_uncertainty_is_not_parsed_as_half_life_uncertainty():
    r=measurement('The half-live is >480(79.6 ±0.2%) min in full mouse serum.',prose=True)
    assert r['half_life_seconds'] is None
    assert r['reported_error_seconds'] is None
    assert r['upper_bound_seconds'] is None
    assert r['lower_bound_seconds']==28800


def test_duration_class_respects_bounds_and_ambiguous_class_sets():
    from proteolysis_data_manager.integration.half_life.labels import class_candidates
    assert class_candidates(measurement('<30','minutes'))==['short']
    assert class_candidates(measurement('<2','hours'))==['short','medium']
    assert class_candidates(measurement('>24','hours'))==['long']
    assert class_candidates(measurement('1','hours'))==['medium']
    assert class_candidates(measurement('6','hours'))==['medium']
    assert class_candidates(measurement('30','minutes'))==['medium']
    assert class_candidates(measurement('>6','hours'))==['long']
    assert class_candidates(measurement('>=6','hours'))==['medium','long']
    assert class_candidates(measurement('<=30','minutes'))==['short','medium']
    assert class_candidates(measurement('~30','minutes'))==[]
    assert class_candidates(measurement('<=1','hours'))==['short','medium']


def test_protocol_defaults_are_traceable_and_dilution_is_not_a_gate():
    from proteolysis_data_manager.protocols.half_life.filtering import prepare_protocol
    from proteolysis_data_manager.integration.half_life.labels import add_duration_classes
    rows=pd.DataFrame([peplife_row(dict(id=str(i),seq='AAA',half_life='30',units_half='minutes')) for i in range(3)])
    rows=add_duration_classes(rows)
    rows['linked_label_status']='not_linked'; rows['is_redundant_representation']=False
    rows['species']='human'; rows['matrix']='serum'; rows['in_vivo_in_vitro']='in vitro'
    rows['ph']=[None,8.,None]; rows['temperature_c']=[None,37,37]
    rows['matrix_fraction']=[.2,1.,None]
    rows['peptide_concentration_raw']=['5 µM']*3
    from proteolysis_data_manager.protocols.half_life.conditions import condition_frame
    rows=condition_frame(rows)
    config={'group':{'species':'human','matrix':'serum','in_vivo_in_vitro':'in vitro'},'defaults':{'ph':7.4,'temperature_c':37},'acceptance_bounds':{'ph':[7.35,7.45],'temperature_c':[36,38]}}
    from proteolysis_data_manager.integration.half_life.schema import normalize_inventory
    compact,_=normalize_inventory(rows)
    result=prepare_protocol(compact,config)
    assert result.ph.isna().tolist()==[True,False,True]
    assert result.ph_effective.tolist()==[7.4,8,7.4]
    assert result.condition_status.tolist()==['accepted','outside','accepted']
    assert result.peptide_per_serum_um.iloc[0]==25
    assert pd.isna(result.peptide_per_serum_um.iloc[2])
    assert result.ci_lower_seconds.isna().all()


def test_protocol_export_retains_all_label_availability_with_compact_contract(tmp_path):
    from proteolysis_data_manager.protocols.half_life.filtering import prepare_protocol, export_protocol
    from proteolysis_data_manager.protocols.half_life.conditions import condition_frame
    from proteolysis_data_manager.integration.half_life.labels import add_duration_classes
    from proteolysis_data_manager.integration.half_life.schema import normalize_inventory, read_csv, CLEAN_COLUMNS
    rows=pd.DataFrame([peplife_row(dict(id=str(i),seq='AXA' if i==5 else 'AAA',half_life=value,units_half='minutes'))
                       for i,value in enumerate(['21 ± 2','<30','~30','Stability increased','N.A.','40','50'])])
    for field,value in {'species':'human','matrix':'serum','in_vivo_in_vitro':'in vitro'}.items(): rows[field]=value
    compact,_=normalize_inventory(condition_frame(add_duration_classes(rows)))
    result=prepare_protocol(compact,protocol_config())
    (tmp_path/'manifests').mkdir();(tmp_path/'manifests/normalization_manifest.json').write_text('{}')
    numeric,classes,folder=export_protocol(result,tmp_path,'peplife2',protocol_config())
    assert len(numeric)==3 and len(classes)==1
    exported=read_csv(folder/'peplife2.csv')
    assert list(exported.columns)==CLEAN_COLUMNS and len(exported)==7
    assert exported.half_life_seconds.isna().sum()==4
    assert exported.ci_lower_seconds.isna().all() and exported.ci_upper_seconds.isna().all()
    assert exported.sd_seconds.isna().all()
    assert numeric.half_life_seconds.iloc[0]==1260
    assert classes.half_life_class.iloc[0]=='short'
    assert exported.loc[5,'sequence']=='AXA'  # Representation is preserved for review.
    assert len(list(folder.glob('*')))==1 and not (tmp_path/'reports').exists()
    assert (tmp_path/'manifests/protocol_human_serum_peplife2.json').is_file()
    assert not (folder.parent/'human_serum.csv').exists()


@pytest.mark.parametrize('prefix',['<','>','approximately '])
def test_censored_or_approximate_plus_minus_is_not_exact(prefix):
    r=measurement('Half-life is '+prefix+'30 ± 2 min in mouse serum.',prose=True)
    assert r['half_life_seconds'] is None


def test_normalization_enriches_shared_measurements_then_aggregation_retains_classes(tmp_path):
    from proteolysis_data_manager.protocols.half_life.filtering import prepare_protocol, export_protocol
    from proteolysis_data_manager.integration.half_life.aggregation import aggregate_environment, merge_observations, source_accounting
    from proteolysis_data_manager.protocols.half_life.conditions import condition_frame
    from proteolysis_data_manager.integration.half_life.labels import add_duration_classes
    from proteolysis_data_manager.integration.half_life.schema import normalize_inventory, source_mask, CLEAN_COLUMNS
    rows=pd.DataFrame([peplife_row(dict(id=str(i),seq='AAA',half_life=value,units_half='minutes')) for i,value in enumerate(['21 ± 2','<30'])])
    for field,value in {'species':'human','matrix':'serum','in_vivo_in_vitro':'in vitro'}.items(): rows[field]=value
    rows['representation_key']=['verified-point','verified-class']
    rows=condition_frame(add_duration_classes(rows))
    duplicate=rows.iloc[[0]].copy();duplicate['observation_id']='peplife2:all:999';duplicate['source_record_id']='999'
    other=rows.copy();other['source']='plifepred';other['observation_id']=['plifepred:natural:0','plifepred:natural:1']
    other['ph']=7.4;other['matrix_fraction']=.25
    compact,audit=normalize_inventory(pd.concat([rows,duplicate,other],ignore_index=True))
    assert len(compact)==2 and compact.source.map(len).eq(2).all()
    assert compact.ph.eq(7.4).all() and compact.matrix_fraction.eq(.25).all()
    assert audit.enriched_fields.map(bool).sum()==3
    assert len(compact.source_record_id.iloc[0])==3
    (tmp_path/'manifests').mkdir();(tmp_path/'manifests/normalization_manifest.json').write_text('{}')
    config=protocol_config()
    for source in ['peplife2','plifepred']:
        export_protocol(prepare_protocol(compact.loc[source_mask(compact,source)],config),tmp_path,source,config)
    export_protocol(prepare_protocol(compact.iloc[:0],config),tmp_path,'cavaco',config)
    incoming,merged,lineage,_,manifest=aggregate_environment(tmp_path,'human_serum',['peplife2','plifepred','cavaco'])
    assert len(incoming)==4 and len(merged)==2 and len(lineage)==5
    assert list(merged.columns)==CLEAN_COLUMNS
    assert merged.ci_lower_seconds.isna().all() and merged.ci_upper_seconds.isna().all()
    assert merged.half_life_seconds.isna().sum()==1
    assert set(merged.half_life_class)=={'short'}
    assert manifest['merged_retained_rows']==2
    assert not (tmp_path/'reports').exists() and not list((tmp_path/'processed').rglob('*.parquet'))
    accounts=source_accounting(tmp_path,compact,incoming,merged,config,['peplife2','plifepred','cavaco'])
    assert accounts['label_types'].numeric.tolist()==[1,1,0]
    assert accounts['label_types'].class_only.tolist()==[1,1,0]
    assert accounts['peplife2_overlap'].set_index('source').loc['plifepred','verified_same_measurements']==2
    changed=incoming.copy();changed.loc[2:,'matrix_fraction']=.5
    assert len(merge_observations(changed)[0])==4  # Dilution differences prevent merging.
    path=tmp_path/'processed/human_serum/sources/plifepred.csv'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='Source CSV changed'):
        aggregate_environment(tmp_path,'human_serum',['peplife2','plifepred'])


def protocol_config():
    return {'group':{'species':'human','matrix':'serum','in_vivo_in_vitro':'in vitro'},'defaults':{'ph':7.4,'temperature_c':37},'acceptance_bounds':{'ph':[7.35,7.45],'temperature_c':[36,38]}}


def test_ordinary_range_vs_documented_ci_and_round_trip(tmp_path):
    from proteolysis_data_manager.integration.half_life.labels import add_duration_classes
    from proteolysis_data_manager.integration.half_life.schema import normalize_inventory, write_parquet, read_parquet, write_csv, read_csv
    raw=pd.DataFrame([peplife_row(dict(id=str(i),seq='AAA',half_life='30-90',units_half='minutes')) for i in range(2)])
    raw=add_duration_classes(raw)
    raw.loc[1,'ci_lower_seconds']=1800;raw.loc[1,'ci_upper_seconds']=5400
    raw.loc[1,'ci_confidence_level']=.95;raw.loc[1,'ci_origin']='explicitly_documented_ci'
    compact,_=normalize_inventory(raw)
    assert pd.isna(compact.half_life_seconds.iloc[0])
    assert compact.half_life_seconds.iloc[1]==3600
    assert compact.half_life_class.tolist()==['medium','medium']
    write_parquet(compact,tmp_path/'data.parquet');write_csv(compact,tmp_path/'data.csv')
    pd.testing.assert_frame_equal(read_parquet(tmp_path/'data.parquet'),compact)
    pd.testing.assert_frame_equal(read_csv(tmp_path/'data.csv'),compact)
    prose=measurement('Half-life is 30–90 min in human serum.',prose=True)
    assert prose['relation']=='interval' and prose['half_life_seconds'] is None


def test_conflicting_aliases_and_independent_equal_sequences_stay_separate():
    from proteolysis_data_manager.integration.half_life.labels import add_duration_classes
    from proteolysis_data_manager.integration.half_life.schema import normalize_inventory
    raw=pd.DataFrame([peplife_row(dict(id=str(i),seq='AAA',half_life='30',units_half='minutes')) for i in range(3)])
    raw=add_duration_classes(raw);raw['representation_key']=['verified','verified','independent']
    raw['matrix_fraction']=[.2,.5,.2]
    compact,audit=normalize_inventory(raw)
    assert len(compact)==3  # Conflict is not averaged; coincidence is not identity.
    assert sum(bool(x) for x in audit.conflicting_fields)==2


def test_remarks_explain_project_heuristic_and_field_donors(tmp_path):
    from proteolysis_data_manager.integration.half_life.labels import add_duration_classes
    from proteolysis_data_manager.integration.half_life.schema import normalize_inventory
    from proteolysis_data_manager.integration.half_life.remarks import write_remarks
    raw=pd.DataFrame([peplife_row(dict(id='2686',seq='AAA',half_life='>1',units_half='hours',pmid='25025626'))])
    raw=add_duration_classes(raw);raw['representation_key']='verified'
    assert raw.duration_class.iloc[0]=='medium'
    compact,audit=normalize_inventory(raw)
    write_remarks(raw,audit,tmp_path)
    note=(tmp_path/'normalized/remarks_peplife2.md').read_text()
    assert 'peplife2:all:2686' in note and 'Project heuristic' in note
    assert '>6 h remains possible' in note and '10.1016/j.biocel.2014.07.004' in note
    assert pd.isna(compact.half_life_seconds.iloc[0])


def test_reviewed_pk_context_does_not_treat_elisa_plate_coating_as_incubation():
    from proteolysis_data_manager.protocols.half_life.conditions import condition_frame
    record=peplife_row(dict(id='5066',seq='AAA',half_life='1.4',units_half='Hours',pmid='33057063',
                           incubation_time='Overnight at 4 °C',vivo_vitro='In Vitro',test_sample='Mouse serum',conc='10 μg/mL'))
    result=condition_frame(pd.DataFrame([record]))
    assert result.in_vivo_in_vitro.iloc[0]=='in vivo'
    assert result.temperature_c.isna().all() and result.peptide_concentration_um.isna().all()
    assert result.half_life_seconds.iloc[0]==5040
    assert 'Overnight at 4' in result.raw_record_json.iloc[0]


def test_time_range_does_not_hide_explicit_incubation_temperature():
    from proteolysis_data_manager.protocols.half_life.conditions import condition_frame
    rows=pd.DataFrame([peplife_row(dict(id=str(i),seq='AAA',half_life='1',units_half='hours',incubation_time=wording))
                       for i,wording in enumerate(['2–24 hours at 37 °C; pH 7.4','25–37 °C; pH 7–8'])])
    result=condition_frame(rows)
    assert result.temperature_c.iloc[0]==37 and result.ph.iloc[0]==7.4
    assert pd.isna(result.temperature_c.iloc[1]) and pd.isna(result.ph.iloc[1])
