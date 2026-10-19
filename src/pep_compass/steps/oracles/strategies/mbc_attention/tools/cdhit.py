from __future__ import annotations

import os
from glob import glob
from random import randint, sample, seed
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tools.base import *
from tools.preProcessed import *

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def executeCdhit(
    input: str | Path, output: str | Path, c: float = 0.4, n: int = 2
) -> str:
    """Execute cdhit.

    :param input: Input.
    :type input: str | Path
    :param output: Output.
    :type output: str | Path
    :param c: C.
    :type c: float
    :param n: N.
    :type n: int
    :return: Result of this operation.
    :rtype: str
    """
    cmd = "cd-hit -c %.1f -n %d -i %s -o %s" % (c, n, input, output)
    msg = os.popen(cmd).read()
    print(msg)
    return msg


def findCluster(clstr_path: str | Path) -> list[str]:
    """Find cluster.

    :param clstr_path: Clstr path.
    :type clstr_path: str | Path
    :return: Result of this operation.
    :rtype: list[str]
    """
    sub_clstr = []
    clstrs = []
    c = 0
    with open(clstr_path, "r") as f:
        for line in f.readlines():
            if not (">Cluster" in line):
                # uniprot: >Cluster 0
                # '0\t26aa, >P37300|ori... at 42.31%'
                # kaliumdb: >Cluster 0
                # 0 223aa, >Q91055... *
                name = line.split("...")[0].split(">")[-1].split("...")[0]
                # print(name)
                sub_clstr.append(name)
            if ">Cluster" in line:
                clstrs.append(sub_clstr)
                sub_clstr = []
        clstrs.append(sub_clstr)
    return clstrs[1:]


# keepTestXpercentToTestAddTrain
def split2sets(clstrs: Sequence[str], x: int = 10) -> tuple[object, object, object]:
    # x = 20/120
    """Split2sets.

    :param clstrs: Clstrs.
    :type clstrs: Sequence[str]
    :param x: X.
    :type x: int
    :return: Result of this operation.
    :rtype: tuple[object, object, object]
    """
    train_names, test_names, novel_names = [], [], []
    seed(1)
    # c = 0
    for sub_clstr in clstrs:
        clstr_no = len(sub_clstr)
        if clstr_no == 1:
            novel_names.extend(sub_clstr)
            # print("novel", novel_names)
        elif clstr_no > 1:
            n_test = int(np.ceil(x / 100 * clstr_no))
            for i in range(n_test):
                rnd_ind = randint(0, clstr_no - 1)
                # print("Cluster: ", c)
                test_name = sub_clstr[rnd_ind]
                test_names.append(test_name)
                sub_clstr.remove(test_name)
                clstr_no -= 1
            train_names.extend(sub_clstr)
    train_names, test_names = keepTestXpercentToTestAddTrain(
        train_names, test_names, x=x
    )
    return train_names, test_names, novel_names


def keepTestXpercentToTestAddTrain(
    train_names: Sequence[str], test_names: Sequence[str], x: int = 20
) -> tuple[list[object], list[object]]:
    """:param train_names: given train names list
    :param test_names: given test names list
    :param x: is the desired_ratio = desired_test_len / (desired_test_len + desired_train_len) * 100;
    real_ratio = test_len / (test_len + train_len) * 100;
    :return: if x == 0 or the given desired_ratio == real_ratio, then do nothing and return train_names and test_names;
    else change train_names and test_names make them to the desired ratio

    :type train_names: Sequence[str]
    :type test_names: Sequence[str]
    :type x: int
    :rtype: tuple[list[object], list[object]]
    """
    test_len, train_len = len(test_names), len(train_names)
    train_names, test_names = np.array(train_names), np.array(test_names)
    test_len_to = int((train_len + test_len) * x / 100)
    if x == 0:
        return train_names.tolist(), test_names.tolist()
    if test_len_to == 0:
        test_len_to = 1
    if test_len_to > test_len:
        add_test_num = test_len_to - test_len
        # select add_test_num from train add them to test
        add_list = sample(list(range(train_len)), add_test_num)
        train_ind = [True] * train_len
        for i in add_list:
            train_ind[i] = False
        new_train = train_names[train_ind]
        rm_train = [train_names[i] for i in add_list]
        new_test = np.append(test_names, rm_train)
    elif test_len_to < test_len:
        rm_test_num = test_len - test_len_to
        # romve rm_test_num test elements and add them to train
        rm_list = sample(list(range(test_len)), rm_test_num)
        test_ind = [True] * test_len
        for i in rm_list:
            test_ind[i] = False
        new_test = test_names[test_ind]
        rm_test = [test_names[i] for i in rm_list]
        new_train = np.append(train_names, rm_test)
    else:
        new_train = train_names
        new_test = test_names
    return new_train.tolist(), new_test.tolist()


def getSeqRecords(
    fasta_path: str | Path,
    train_names: Sequence[str],
    test_names: Sequence[str],
    novel_names: Sequence[str],
) -> tuple[object, object, object]:
    """Get seq records.

    :param fasta_path: Fasta path.
    :type fasta_path: str | Path
    :param train_names: Train names.
    :type train_names: Sequence[str]
    :param test_names: Test names.
    :type test_names: Sequence[str]
    :param novel_names: Novel names.
    :type novel_names: Sequence[str]
    :return: Result of this operation.
    :rtype: tuple[object, object, object]
    """
    train_set, test_set, novel_set = [], [], []
    for record in SeqIO.parse(fasta_path, "fasta"):
        name = record.name
        if name in novel_names:
            novel_set.append(record)
        if name in test_names:
            test_set.append(record)
        if name in train_names:
            train_set.append(record)
    return train_set, test_set, novel_set


def generateTrainTestNovelDataset(
    fasta_path: str | Path,
    out_fasta_path: str | Path,
    x: int = 10,
    del_novel: bool = True,
) -> tuple[str, str, str] | tuple[str, str, str, str]:
    """Generate train test novel dataset.

    :param fasta_path: Fasta path.
    :type fasta_path: str | Path
    :param out_fasta_path: Out fasta path.
    :type out_fasta_path: str | Path
    :param x: X.
    :type x: int
    :param del_novel: Del novel.
    :type del_novel: bool
    :return: Result of this operation.
    :rtype: tuple[str, str, str] | tuple[str, str, str, str]
    """
    root_dir = getRootDir()
    fasta_name = os.path.basename(out_fasta_path)
    clstr_path = out_fasta_path + ".clstr"
    folder = os.path.dirname(out_fasta_path)
    act_name, suffix = fasta_name.split(".")
    novel_path = os.path.join(folder, "novel_" + fasta_name)
    if del_novel:
        all_path = os.path.join(folder, "delNovel_" + fasta_name)
    else:
        test_path = os.path.join(folder, "test_" + fasta_name)
        train_path = os.path.join(folder, "train_" + fasta_name)

    clstrs = findCluster(clstr_path)
    train_names, test_names, novel_names = split2sets(clstrs, x=x)
    train_set, test_set, novel_set = getSeqRecords(
        fasta_path, train_names, test_names, novel_names
    )
    SeqIO.write(novel_set, novel_path, "fasta")
    msg = "The dataset was split into 3 subsets by clstr file, they are: "
    msg += "\n\t%s" % novel_path
    msg += "\n\tsequence number: %d" % len(novel_set)
    if del_novel:
        train_set.extend(test_set)
        SeqIO.write(train_set, all_path, "fasta")
        msg += "\n\t%s" % all_path
        msg += "\n\tsequence number: %d" % len(train_set)
        print(msg)
        return msg, novel_path, all_path
    else:
        SeqIO.write(test_set, test_path, "fasta")
        SeqIO.write(train_set, train_path, "fasta")
        msg += "\n\t%s" % test_path
        msg += "\n\tsequence number: %d" % len(test_set)
        msg += "\n\t%s" % train_path
        msg += "\n\tsequence number: %d" % len(train_set)
        print(msg)
        return msg, novel_path, test_path, train_path


# the fasta folder is the folder corresponding to ion-channel root folder
# fasta_folder = 'uniprot-preprocessed-dataset'
# fasta_name = 'calcium.fasta'
def executeCdhitAndSplitDataset(
    fasta_path: str | Path,
    c: float = 0.4,
    n: int = 2,
    x: int = 10,
    del_novel: bool = False,
) -> tuple[object, str, object, object] | tuple[object, str, object, object, object]:
    """Execute cdhit and split dataset.

    :param fasta_path: Fasta path.
    :type fasta_path: str | Path
    :param c: C.
    :type c: float
    :param n: N.
    :type n: int
    :param x: X.
    :type x: int
    :param del_novel: Del novel.
    :type del_novel: bool
    :return: Result of this operation.
    :rtype: tuple[object, str, object, object] | tuple[object, str, object, object, object]
    """
    root_dir = getRootDir()
    fasta_folder = os.path.dirname(fasta_path)
    fasta_name = os.path.basename(fasta_path)
    folder = fasta_folder
    # clstr_path = os.path.join(folder, clstr_name)
    out_folder_name = "c%dn%d" % (c * 10, n)
    # set output fasta name after cd hit process
    output_fasta = fasta_name.split(".")[0] + "_cdhit_" + out_folder_name + ".fasta"
    # gene clstr name of the corresponding fasta  after cd hit process
    output_clstr = output_fasta + ".clstr"
    out_folder = os.path.join(root_dir, "data/temp")

    # create output folder of "split_dataset"
    createFolder(out_folder)

    out_fasta_path = os.path.join(out_folder, output_fasta)

    msg = executeCdhit(fasta_path, out_fasta_path, c=c, n=n)

    # generate 3 dataset from the given fasta_path and the corresponding clstr file
    # !!!! read this and recode again there is something wrong in file paths
    if del_novel:
        temp_msg, novel_path, all_path = generateTrainTestNovelDataset(
            fasta_path, out_fasta_path, x=x, del_novel=del_novel
        )
        msg += temp_msg
        return msg, out_folder, novel_path, all_path
    else:
        temp_msg, novel_path, test_path, train_path = generateTrainTestNovelDataset(
            fasta_path, out_fasta_path, x=x, del_novel=del_novel
        )
        msg += temp_msg
        return msg, out_folder, novel_path, test_path, train_path


def cdHitByInput(
    abs_input: str | Path, c: float = 0.8, n: int = 2, redo: bool = True
) -> str:
    """Cd hit by input.

    :param abs_input: Abs input.
    :type abs_input: str | Path
    :param c: C.
    :type c: float
    :param n: N.
    :type n: int
    :param redo: Redo.
    :type redo: bool
    :return: Result of this operation.
    :rtype: str
    """
    in_folder = "/".join(abs_input.split("/")[:-1])
    in_name = abs_input.split("/")[-1].split(".")[0]
    out_folder = "cd-hit-output/c%dn%d" % (c * 10, n)
    out_name = "%s_c%dn%d.fasta" % (in_name, c * 10, n)
    abs_out_folder = os.path.join(in_folder, out_folder)
    createFolder(abs_out_folder)
    abs_output = os.path.join(abs_out_folder, out_name)
    msg = ""
    path_flag = os.path.isfile(abs_output)
    if (path_flag and redo) or (not path_flag):
        if path_flag:
            print(
                "%s\n\talready existed and it will be replaced by cdHitByInput\n"
                % abs_output
            )
        else:
            print(
                "%s\n\tnot existed and it will be generated by cdHitByInput\n"
                % abs_output
            )
        msg += executeCdhit(abs_input, abs_output, c=c, n=n)
    else:
        print(
            "%s\n\talready existed and nothing will be implement by cdHitByInput\n"
            % abs_output
        )

    print(msg)
    return abs_output


def executeCdhit2d(
    pos_in: str | Path, neg_in: str | Path, neg_out: str | Path, c: int = 1, n: int = 2
) -> str:
    """Execute cdhit2d.

    :param pos_in: Pos in.
    :type pos_in: str | Path
    :param neg_in: Neg in.
    :type neg_in: str | Path
    :param neg_out: Neg out.
    :type neg_out: str | Path
    :param c: C.
    :type c: int
    :param n: N.
    :type n: int
    :return: Result of this operation.
    :rtype: str
    """
    path_flag = os.path.isfile(neg_out)
    msg = ""
    cmd = "cd-hit-2d -c %.1f -n %d -i %s -i2 %s -o %s" % (c, n, pos_in, neg_in, neg_out)
    msg += os.popen(cmd).read()
    return msg


def fastas2Records(
    fastas: Sequence[Sequence[str]],
    replaced: bool = True,
    remove_dup_shortSeq: bool = False,
    shorest_len: int = 5,
    largerest_len: int = 60,
) -> tuple[list[object], list[object], object]:
    """input fastas and output
    1. fasta files (>10 seqs) and its corresponding records for cdhit
    2. short_fastas (len in range of [6,10])
    3. delete the duplicated sequences among all the sequences of short_fastas
    4. all the sequences contain unnatural a.a.s are removed
    :param fastas: fastas with n samples is a list like: [[ID0, Seq0], [ID1, Seq1], ..., [IDn, Seqn]]
    :param replaced: True when one want to replace the previous same path file
    :return: 1. records (>10 seqs)
    2. short_fastas ([6,10] unique seqs)
    3. fasta file path of records (>10 seqs things of 1.)

    :type fastas: Sequence[Sequence[str]]
    :type replaced: bool
    :param remove_dup_shortSeq: Remove dup shortseq.
    :type remove_dup_shortSeq: bool
    :param shorest_len: Shorest len.
    :type shorest_len: int
    :param largerest_len: Largerest len.
    :type largerest_len: int
    :rtype: tuple[list[object], list[object], object]
    """
    root_dir = getRootDir()
    out_dir = os.path.join(root_dir, "data", "temp")
    createFolder(out_dir)
    out_path = os.path.join(out_dir, "temp.fasta")
    if not replaced:
        out_path = getUnexistedName(out_path, ".fasta")
    records = []
    # short_records = []
    short_fastas = []
    short_seqs = []
    lens = []
    for i in range(len(fastas)):
        fasta = fastas[i]
        mic = "_".join([str(i) for i in fasta[2:]])
        seq = fasta[1]
        if any((s in unnatural_amino_acids) for s in seq):
            continue
        l = len(fasta[1])
        lens.append(l)
        if l >= shorest_len and l <= largerest_len:
            name = str(fasta[0])
            record = SeqRecord(Seq(seq), name, name, mic)
            if l > 10:
                records.append(record)
            else:
                if remove_dup_shortSeq:
                    # remove same seqs of short seqs
                    if not (seq in short_seqs):
                        short_seqs.append(seq)
                        short_fastas.append(fasta)
                else:
                    short_seqs.append(seq)
                    short_fastas.append(fasta)
    SeqIO.write(records, out_path, "fasta")
    print("shortest Length: %d" % np.min(lens))
    return records, short_fastas, out_path


def readfastasFromFastaPath(fasta_path: str | Path) -> list[object]:
    """Readfastas from fasta path.

    :param fasta_path: Fasta path.
    :type fasta_path: str | Path
    :return: Result of this operation.
    :rtype: list[object]
    """
    fastas = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        mic = record.description
        mic = mic.split(" ")[-1]
        # print("mic", mic)
        if "_" in mic:
            mics = mic.split("_")
            # print("mics", mics)
            fst = [record.name, str(record.seq)]
            for m in mics:
                fst.append(float(m))
            fastas.append(fst)
        else:
            fastas.append([record.name, str(record.seq), float(mic)])
    # fastas = pd.DataFrame(fastas, columns=colnames)
    return fastas


def cdhitByfastas(
    fastas: Sequence[Sequence[str]], thr: int = 1
) -> Sequence[Sequence[str]]:
    """Cdhit byfastas.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param thr: Thr.
    :type thr: int
    :return: Result of this operation.
    :rtype: Sequence[Sequence[str]]
    """
    records, short_fastas, path = fastas2Records(fastas, thr)
    out_path = "%s-cdhit_thr%.2f.fasta" % (path.split(".fasta")[0], thr)
    msg = executeCdhit(path, out_path, c=thr)
    fastas = readfastasFromFastaPath(out_path)
    # whole_fastas = [fastas.append(i) for i in short_fastas]
    fastas.extend(short_fastas)
    return fastas


def fastas2DeleteFasta1ByCdhit2d(
    fastas1: Sequence[Sequence[str]], fastas2: Sequence[Sequence[str]], thr: int = 0
) -> np.ndarray:
    """Fastas2delete fasta1by cdhit2d.

    :param fastas1: Fastas1.
    :type fastas1: Sequence[Sequence[str]]
    :param fastas2: Fastas2.
    :type fastas2: Sequence[Sequence[str]]
    :param thr: Thr.
    :type thr: int
    :return: Result of this operation.
    :rtype: np.ndarray
    """
    records1, short_fastas1, path1 = fastas2Records(fastas1, replaced=True)
    records2, short_fastas2, path2 = fastas2Records(fastas2, replaced=False)
    out_path = "%s-cdhit2d_thr%.2f.fasta" % (path2.split(".fasta")[0], thr)
    msg = executeCdhit2d(path1, path2, out_path)
    fastas = readfastasFromFastaPath(out_path)
    short_fastas = []
    for fasta in short_fastas2:
        if not (fasta in short_fastas1):
            short_fastas.append(fasta)
    fastas.extend(short_fastas)
    return fastas


def removeSpecieDuplicatedSeq(specie_name: str, thr: int = 1) -> str:
    """please do mrg_path = removeMergeDuplicatedSeq(thr=thr) before execute this function
    1. remove itself's duplicated sequences
    2. remove sequeces the same with merge sequeces

    :param specie_name: Specie name.
    :type specie_name: str
    :param thr: Thr.
    :type thr: int
    :return: Result of this operation.
    :rtype: str
    """
    name = specie_name.replace(" ", "-")
    fld = GetOriFolder()
    spe = GetOriSpecieInfo(specie_name)
    mrg = GetOriMergeInfo()
    # remove itself's duplicated sequences
    if thr == 0:
        fastas1 = mrg.mrg_fastas
        fastas2 = spe.fastas
        data_dir = os.path.join(fld.root_dir, "data")
    else:
        fastas1 = cdhitByfastas(mrg.mrg_fastas, thr)
        fastas2 = cdhitByfastas(spe.fastas, thr)
        data_dir = os.path.join(fld.root_dir, "data", "cdhit%.1f" % thr)
    # remove sequeces the same with merge sequeces
    fastas = fastas2DeleteFasta1ByCdhit2d(fastas1, fastas2, thr)
    createFolder(data_dir)
    specie_path = os.path.join(data_dir, "specie_%s.csv" % name)
    df = pd.DataFrame(fastas, columns=["ID", "SEQUENCE", "pMIC"])
    df.to_csv(specie_path, index=False)
    print("Snapshot preprocessed file to %s" % specie_path)
    return specie_path


def removeMergeDuplicatedSeq(thr: int = 1) -> str:
    """Remove merge duplicated seq.

    :param thr: Thr.
    :type thr: int
    :return: Result of this operation.
    :rtype: str
    """
    fld = GetOriFolder()
    mrg = GetOriMergeInfo()
    whole_fastas = cdhitByfastas(mrg.mrg_fastas, thr)
    name = os.path.basename(mrg.mrg_path)
    mrg_dir = os.path.join(fld.root_dir, "data", "merge", "cdhit%.1f" % thr)
    createFolder(mrg_dir)
    mrg_path = os.path.join(mrg_dir, name)
    df = pd.DataFrame(whole_fastas, columns=["ID", "SEQUENCE", "pMIC"])
    df.to_csv(mrg_path, index=False)
    print("Snapshot preprocessed file to %s" % mrg_path)
    return mrg_path


def SplitMerge2TrainTestNovel(
    host_specie: str = "escherichia coli",
    target_specie: str = "staphylococcus aureus",
    thr: int = 0,
) -> dict[object, object]:
    """Split merge2train test novel.

    :param host_specie: Host specie.
    :type host_specie: str
    :param target_specie: Target specie.
    :type target_specie: str
    :param thr: Thr.
    :type thr: int
    :return: Result of this operation.
    :rtype: dict[object, object]
    """
    mrg = GetOriMergeInfo(host_specie, target_specie)
    colnames = mrg.colnames
    fastas = mrg.mrg_fastas.to_numpy().tolist()
    records, short_fastas, path = fastas2Records(fastas, thr)
    msg, out_folder, novel_path, test_path, train_path = executeCdhitAndSplitDataset(
        path, del_novel=False
    )
    fastas = {}
    fastas["mrg"] = mrg.mrg_fastas
    fastas["train"] = readfastasFromFastaPath(train_path)
    fastas["test"] = readfastasFromFastaPath(test_path)
    novel_fastas = readfastasFromFastaPath(novel_path)
    novel_fastas.extend(short_fastas)
    fastas["novel"] = novel_fastas
    name = os.path.basename(mrg.mrg_path)
    fld1 = GetFolder()
    mrg_dir = fld1.data_dir
    createFolder(mrg_dir)
    paths = {}
    path = os.path.join(mrg_dir, name)
    paths["mrg"] = path
    df = pd.DataFrame(fastas["mrg"], columns=colnames)
    df.to_csv(path, index=False)
    print("Snapshot preprocessed file to %s" % path)
    for i in ["train", "test", "novel"]:
        path = os.path.join(mrg_dir, "%s_%s" % (i, name))
        paths[i] = path
        df = pd.DataFrame(fastas[i], columns=colnames)
        df.to_csv(path, index=False)
        print("Snapshot preprocessed file to %s" % path)
    return paths


# < 40% identity to novel, others randomly select 10% as test,
# remained select 10% as val-train, last 90% as tra-train
def SplitSpecie2TrainValTestNovel(
    specie: str = "escherichia coli", thr: int = 0, cdhit_flag: bool = False
) -> dict[object, object]:
    """Split specie2train val test novel.

    :param specie: Specie.
    :type specie: str
    :param thr: Thr.
    :type thr: int
    :param cdhit_flag: Cdhit flag.
    :type cdhit_flag: bool
    :return: Result of this operation.
    :rtype: dict[object, object]
    """
    spe = GetOriSpecieInfo(specie)
    colnames, abb = spe.colnames, spe.specie_abb
    fastas = spe.fastas.to_numpy().tolist()
    records, short_fastas, path = fastas2Records(fastas, thr)
    fastas = {}
    if cdhit_flag:
        msg, out_folder, novel_path, test_cdhit_path, train_cdhit_path = (
            executeCdhitAndSplitDataset(path, del_novel=False)
        )
        test_cdhit_fastas = readfastasFromFastaPath(test_cdhit_path)
        train_cdhit_fastas = readfastasFromFastaPath(train_cdhit_path)
        fastas["test_cdhit"] = test_cdhit_fastas
        fastas["train_cdhit"] = train_cdhit_fastas
        fasta_names = ["train_cdhit", "test_cdhit"]
    else:
        msg, out_folder, novel_path, all_path = executeCdhitAndSplitDataset(
            path, del_novel=True
        )
        # fastas["spe"] = spe.fastas
        fastas["delNovel"] = readfastasFromFastaPath(all_path)
        fasta_names = ["delNovel", "novel"]
    # fastas["test"] = readfastasFromFastaPath(test_path)
    novel_fastas = readfastasFromFastaPath(novel_path)
    novel_fastas.extend(short_fastas)
    fastas["novel"] = novel_fastas
    name = "%s.csv" % abb
    fld1 = GetFolder()
    mrg_dir = fld1.data_dir
    createFolder(mrg_dir)
    paths = {}
    # path = os.path.join(mrg_dir, name)
    # paths["spe"] = path
    # df = pd.DataFrame(fastas["spe"], columns=colnames)
    # df.to_csv(path, index=False)
    print("Snapshot preprocessed file to %s" % path)
    for i in fasta_names:
        path = os.path.join(mrg_dir, "%s_%s" % (i, name))
        paths[i] = path
        df = pd.DataFrame(fastas[i], columns=colnames)
        df.to_csv(path, index=False)
        print("Snapshot preprocessed file to %s" % path)

    if cdhit_flag:
        tra_cdhit_path, val_cdhit_path = splitValidationFromTrain(paths["train_cdhit"])
        paths["%s-train" % abb] = train_cdhit_path
        paths["%s-test" % abb] = test_cdhit_path
        paths["%s-tra-train" % abb] = tra_cdhit_path
        paths["%s-val-train" % abb] = val_cdhit_path
    else:
        print("saved %s fastas to: \n\t%s" % (specie, paths["delNovel"]))
        train_path, test_path = splitTestFromAll(paths["delNovel"])
        tra_path, val_path = splitValidationFromTrain(train_path)
        paths["%s-train" % abb] = train_path
        paths["%s-test" % abb] = test_path
        paths["%s-tra-train" % abb] = tra_path
        paths["%s-val-train" % abb] = val_path
    return paths
