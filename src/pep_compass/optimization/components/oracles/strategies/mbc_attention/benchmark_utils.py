from Bio import SeqIO
import numpy as np
import pandas as pd
from tensorflow import keras
from typing import List, Tuple

import config
from tools.MultiBranchCNN import CNNimportFtsDataSetsPoNe, CNNstandardInputOutput


def convert_inputs(input_file: str, features: List = config.def_fts
) -> Tuple[List, pd.DataFrame]:
    """Convert inputs for the MBC-Attention model.

    Args:
        input_file: path to the input fasta file.
        features: names of sequence-derived descriptors used for inference.

    Returns:
        A tuple (processed_inputs, sequence_df) where processed_inputs is a
        list of featurized protein sequences from the input file; while
        sequence_df is a data frame containing information about proteins
        ids and sequences before conversion.
    """
    print("Converting inputs")
    names, sequences = [], []
    for sequence in SeqIO.parse(input_file, "fasta"):
        if len(str(sequence.seq)) >= config.min_seq_length and len(str(sequence.seq)) <= config.max_seq_length:
            names.append(sequence.id)
            sequences.append(str(sequence.seq))

    sequence_df = pd.DataFrame.from_dict(
        {"ID": names, "SEQUENCE": sequences}
    )

    # Assume no target MIC values during inference.
    processed_inputs = CNNimportFtsDataSetsPoNe(
        sequence_df, ft_list=features, target=None
    )
    processed_inputs, _ = CNNstandardInputOutput(processed_inputs)

    if len(processed_inputs[0]) != len(sequences):
        raise ValueError(f"Number of sequences before and after processing "
                         f"does not match ({len(sequences)} vs "
                         f"{len(processed_inputs[0])})")

    print("Finished converting inputs")
    return processed_inputs, sequence_df

def run_inference(processed_inputs, sequence_df: pd.DataFrame,
                  model_file: str = "model/whole_train.mdl") -> pd.DataFrame:
    """Run inference for MBC-Attention model.

    Args:
        processed_inputs: input sequences after conversion.
        sequence_df: sequence-id handler for input proteins.
        model_file: path to the MBC-Attention.mdl file.
    """
    print("Running inference")
    model = keras.models.load_model(model_file, compile=False)
    # preds = model.predict_on_batch(processed_inputs)
    preds = model.predict(processed_inputs, batch_size=128)
    preds = preds / config.def_scale - config.def_bias

    sequence_df["Log10_MIC"] = -preds
    sequence_df["MIC"] = 10 ** (-preds)
    print("Finished running inference")
    return sequence_df

def convert_outputs(pred_df: pd.DataFrame, output_file: str) -> None:
    """Convert MBC-Attention output.

    Output conversion follows BattleAMP-benchmark standard.

    Args:
        pred_df: data frame containing predictions as well as protein's
            id and sequence.
        output_file: path to the output file.
    """
    print("Converting outputs")
    pred_df.rename(
        columns={"ID": "Sequence_id", "SEQUENCE": "Sequence"},
        inplace=True
    )
    pred_df = pred_df.assign(MIC_unit="uM")

    pred_df.to_csv(output_file, sep="\t", index=False)
    print("Finished converting outputs")
