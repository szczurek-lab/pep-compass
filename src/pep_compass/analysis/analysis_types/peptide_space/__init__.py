"""Latent-space locality of HydrAMP's own training peptide corpus.

Unlike ``analysis_types.locality``, this does not operate on
``ExperimentSelection``/experiment run output -- it works directly from the
raw peptide corpus (``assets/peptides_data/peptides.csv``) and a HydrAMP
encoder, so it is not wired into ``ExperimentAnalysis``. Import and call
directly:

    from pep_compass.analysis.analysis_types.peptide_space import (
        group_locality,
        load_single_label_groups,
    )
"""

from pep_compass.analysis.analysis_types.peptide_space.group_locality import (
    between_group_distance,
    group_locality,
    load_amp_mic_groups,
    load_single_label_groups,
)

__all__ = [
    "between_group_distance",
    "group_locality",
    "load_amp_mic_groups",
    "load_single_label_groups",
]
