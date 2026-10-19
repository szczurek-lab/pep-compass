import pytest

"""Lazy experiment selection uses the same tables as the run saver."""
from pep_compass.analysis.reader import ExperimentReader
from tests.fixtures.orchestration import create


def test_collection_selection_and_table_handle(tmp_path):
    for seed in (7, 9):
        result = create().run(["AA", "CC"], seed=seed)
        result.save(tmp_path / str(seed))
        result.close()
    reader = ExperimentReader(tmp_path)
    assert len(reader.runs) == 2
    selected = reader.select(seeds=[7])
    assert selected.count_rows("evaluation_archive") == 2
    assert selected.rows(sequence="AA").collect("evaluation_archive")[
        "sequence"
    ].tolist() == ["AA"]
    assert reader.dataset.tables["evaluation_archive"].count_rows() == 4
    with reader.replay(reader.runs[0].run_id) as run:
        assert len(run.get_archive()) == 2


@pytest.mark.parametrize("mode", ["sequential", "batched"])
def test_trajectory_metadata_and_latents_use_one_aligned_payload(tmp_path, mode):
    import torch

    from pep_compass import Pipeline
    from pep_compass.steps.flow import LocalEnumeration
    from pep_compass.steps.oracles import Hydrophobicity
    from tests.fixtures.direct_components import PairGenerator, RandomWalker
    from tests.fixtures.orchestration import FixtureResource

    graph = Pipeline(
        [
            LocalEnumeration(
                walker=RandomWalker(),
                mutation_generator=PairGenerator(),
                iterations=2,
                trajectories=2,
                include_walk_points=False,
                trajectory_execution=mode,
            ),
            Hydrophobicity(),
        ],
        autoencoder=FixtureResource(),
    )
    result = graph.run(
        ["AA"],
        seed=7,
        storage={"reconstruction": "trajectories"},
    )
    points, latents = result.reader.get_trajectories()
    assert len(points) == 4
    assert latents.shape == (4, 2)
    assert points["latent_index"].tolist() == list(range(4))
    assert points.groupby("trajectory_id")["trajectory_step"].apply(list).tolist() == [
        [0, 1],
        [0, 1],
    ]
    assert torch.isfinite(latents).all()
    result.close()
