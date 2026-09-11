import os
from glob import glob
import pandas as pd
import numpy as np
from random import seed, randint, sample
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from tools.base import *
from tools.preProcessed import *

def executeCdhit(input, output, c=0.4, n=2):
    cmd = 'cd-hit -c %.1f -n %d -i %s -o %s' % (c, n, input, output)
    msg = os.popen(cmd).read()
    print(msg)
    return msg

def findCluster(clstr_path):
    sub_clstr = []
    clstrs = []
    c = 0
    with open(clstr_path, "r") as f:
        for line in f.readlines():
            if not(">Cluster" in line):
                # uniprot: >Cluster 0
                # '0\t26aa, >P37300|ori... at 42.31%'
                # kaliumdb: >Cluster 0
                # 0	223aa, >Q91055... *
                name = line.split('...')[0].split(">")[-1].split('...')[0]
                # print(name)
                sub_clstr.append(name)
            if ">Cluster" in line:
                clstrs.append(sub_clstr)
                sub_clstr = []
        clstrs.append(sub_clstr)
    return clstrs[1:]

# keepTestXpercentToTestAddTrain
def split2sets(clstrs, x=10):
    # x = 20/120
    train_names, test_names, novel_names = [], [], []
    seed(1)
    # c = 0
    for sub_clstr in clstrs:
        clstr_no = len(sub_clstr)
        if clstr_no == 1:
            novel_names.extend(sub_clstr)
            # print("novel", novel_names)
        elif clstr_no > 1:
            n_test = int(np.ceil(x/100 * clstr_no))
            for i in range(n_test):
                rnd_ind = randint(0, clstr_no-1)
                # print("Cluster: ", c)
                test_name = sub_clstr[rnd_ind]
                test_names.append(test_name)
                sub_clstr.remove(test_name)
                clstr_no -= 1
            train_names.extend(sub_clstr)
    train_names, test_names = keepTestXpercentToTestAddTrain(train_names, test_names, x=x)
    return train_names, test_names, novel_names

def keepTestXpercentToTestAddTrain(train_names, test_names, x=20):
    """
    :param train_names: given train names list
    :param test_names: given test names list
    :param x: is the desired_ratio = desired_test_len / (desired_test_len + desired_train_len) * 100;
    real_ratio = test_len / (test_len + train_len) * 100;
    :return: if x == 0 or the given desired_ratio == real_ratio, then do nothing and return train_names and test_names;
    else change train_names and test_names make them to the desired ratio
    """
    test_len, train_len = len(test_names), len(train_names)
    train_names, test_names = np.array(train_names), np.array(test_names)
    test_len_to = int((train_len + test_len) * x /100)
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

def getSeqRecords(fasta_path, train_names, test_names, novel_names):
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

def generateTrainTestNovelDataset(fasta_path, out_fasta_path, x=10, del_novel=True):
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
    train_set, test_set, novel_set = getSeqRecords(fasta_path, train_names, test_names, novel_names)
    SeqIO.write(novel_set, novel_path, 'fasta')
    msg = "The dataset was split into 3 subsets by clstr file, they are: "
    msg += "\n\t%s" % novel_path
    msg += "\n\tsequence number: %d" % len(novel_set)
    if del_novel:
        train_set.extend(test_set)
        SeqIO.write(train_set, all_path, 'fasta')
        msg += "\n\t%s" % all_path
        msg += "\n\tsequence number: %d" % len(train_set)
        print(msg)
        return msg, novel_path, all_path
    else:
        SeqIO.write(test_set, test_path, 'fasta')
        SeqIO.write(train_set, train_path, 'fasta')
        msg += "\n\t%s" % test_path
        msg += "\n\tsequence number: %d" % len(test_set)
        msg += "\n\t%s" % train_path
        msg += "\n\tsequence number: %d" % len(train_set)
        print(msg)
        return msg, novel_path, test_path, train_path

# the fasta folder is the folder corresponding to ion-channel root folder
# fasta_folder = 'uniprot-preprocessed-dataset'
# fasta_name = 'calcium.fasta'
def executeCdhitAndSplitDataset(fasta_path, c=0.4, n=2, x=10, del_novel=False):
    root_dir = getRootDir()
    fasta_folder = os.path.dirname(fasta_path)
    fasta_name = os.path.basename(fasta_path)
    folder = fasta_folder
    #clstr_path = os.path.join(folder, clstr_name)
    out_folder_name = "c%dn%d" % (c*10, n)
    # set output fasta name after cd hit process
    output_fasta = fasta_name.split('.')[0] + '_cdhit_' + out_folder_name + '.fasta'
    # gene clstr name of the corresponding fasta  after cd hit process
    output_clstr = output_fasta + '.clstr'
    out_folder = os.path.join(root_dir, "data/temp")

    # create output folder of "split_dataset"
    createFolder(out_folder)

    out_fasta_path = os.path.join(out_folder, output_fasta)

    msg = executeCdhit(fasta_path, out_fasta_path, c=c, n=n)

    # generate 3 dataset from the given fasta_path and the corresponding clstr file
    # !!!! read this and recode again there is something wrong in file paths
    if del_novel:
        temp_msg, novel_path, all_path = generateTrainTestNovelDataset(fasta_path, out_fasta_path, x=x, del_novel=del_novel)
        msg += temp_msg
        return msg, out_folder, novel_path, all_path
    else:
        temp_msg, novel_path, test_path, train_path = generateTrainTestNovelDataset(fasta_path, out_fasta_path, x=x, del_novel=del_novel)
        msg += temp_msg
        return msg, out_folder, novel_path, test_path, train_path


def cdHitByInput(abs_input, c=0.8, n=2, redo=True):
    in_folder = "/".join(abs_input.split("/")[:-1])
    in_name = abs_input.split("/")[-1].split(".")[0]
    out_folder = "cd-hit-output/c%dn%d" % (c*10, n)
    out_name = "%s_c%dn%d.fasta" % (in_name, c*10, n)
    abs_out_folder = os.path.join(in_folder,out_folder)
    createFolder(abs_out_folder)
    abs_output = os.path.join(abs_out_folder, out_name)
    msg = ''
    path_flag = os.path.isfile(abs_output)
    if (path_flag and redo) or (not path_flag):
        if path_flag:
            print("%s\n\talready existed and it will be replaced by cdHitByInput\n" % abs_output)
        else:
            print("%s\n\tnot existed and it will be generated by cdHitByInput\n" % abs_output)
        msg += executeCdhit(abs_input, abs_output, c=c, n=n)
    else:
        print("%s\n\talready existed and nothing will be implement by cdHitByInput\n" % abs_output)

    print(msg)
    return abs_output

def executeCdhit2d(pos_in, neg_in, neg_out, c=1, n=2):
    path_flag = os.path.isfile(neg_out)
    msg = ''
    cmd = 'cd-hit-2d -c %.1f -n %d -i %s -i2 %s -o %s' % (c, n, pos_in, neg_in, neg_out)
    msg += os.popen(cmd).read()
    return msg

def fastas2Records(fastas, replaced=True, remove_dup_shortSeq=False, shorest_len=5, largerest_len=60):
    """
    input fastas and output
    1. fasta files (>10 seqs) and its corresponding records for cdhit
    2. short_fastas (len in range of [6,10])
    3. delete the duplicated sequences among all the sequences of short_fastas
    4. all the sequences contain unnatural a.a.s are removed
    :param fastas: fastas with n samples is a list like: [[ID0, Seq0], [ID1, Seq1], ..., [IDn, Seqn]]
    :param replaced: True when one want to replace the previous same path file
    :return: 1. records (>10 seqs)
    2. short_fastas ([6,10] unique seqs)
    3. fasta file path of records (>10 seqs things of 1.)
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
                    if not(seq in short_seqs):
                        short_seqs.append(seq)
                        short_fastas.append(fasta)
                else:
                    short_seqs.append(seq)
                    short_fastas.append(fasta)
    SeqIO.write(records, out_path, "fasta")
    print("shortest Length: %d" % np.min(lens))
    return records, short_fastas, out_path

def readfastasFromFastaPath(fasta_path):
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

def cdhitByfastas(fastas, thr=1):
    records, short_fastas, path = fastas2Records(fastas, thr)
    out_path = "%s-cdhit_thr%.2f.fasta" % (path.split(".fasta")[0], thr)
    msg = executeCdhit(path, out_path, c=thr)
    fastas = readfastasFromFastaPath(out_path)
    # whole_fastas = [fastas.append(i) for i in short_fastas]
    fastas.extend(short_fastas)
    return fastas

def fastas2DeleteFasta1ByCdhit2d(fastas1, fastas2, thr=0):
    records1, short_fastas1, path1 = fastas2Records(fastas1, replaced=True)
    records2, short_fastas2, path2 = fastas2Records(fastas2, replaced=False)
    out_path = "%s-cdhit2d_thr%.2f.fasta" % (path2.split(".fasta")[0], thr)
    msg = executeCdhit2d(path1, path2, out_path)
    fastas = readfastasFromFastaPath(out_path)
    short_fastas = []
    for fasta in short_fastas2:
        if not(fasta in short_fastas1):
            short_fastas.append(fasta)
    fastas.extend(short_fastas)
    return fastas

def removeSpecieDuplicatedSeq(specie_name, thr=1):
    """
    please do mrg_path = removeMergeDuplicatedSeq(thr=thr) before execute this function
    1. remove itself's duplicated sequences
    2. remove sequeces the same with merge sequeces
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
    print("Save preprocessed file to %s" % specie_path)
    return specie_path

def removeMergeDuplicatedSeq(thr=1):
    fld = GetOriFolder()
    mrg = GetOriMergeInfo()
    whole_fastas = cdhitByfastas(mrg.mrg_fastas, thr)
    name = os.path.basename(mrg.mrg_path)
    mrg_dir = os.path.join(fld.root_dir, "data", "merge", "cdhit%.1f" % thr)
    createFolder(mrg_dir)
    mrg_path = os.path.join(mrg_dir, name)
    df = pd.DataFrame(whole_fastas, columns=["ID", "SEQUENCE", "pMIC"])
    df.to_csv(mrg_path, index=False)
    print("Save preprocessed file to %s" % mrg_path)
    return mrg_path

def SplitMerge2TrainTestNovel(host_specie="escherichia coli", target_specie="staphylococcus aureus", thr=0):
    mrg = GetOriMergeInfo(host_specie, target_specie)
    colnames = mrg.colnames
    fastas = mrg.mrg_fastas.to_numpy().tolist()
    records, short_fastas, path = fastas2Records(fastas, thr)
    msg, out_folder, novel_path, test_path, train_path = executeCdhitAndSplitDataset(path, del_novel=False)
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
    print("Save preprocessed file to %s" % path)
    for i in ["train", "test", "novel"]:
        path = os.path.join(mrg_dir, "%s_%s" % (i, name))
        paths[i] = path
        df = pd.DataFrame(fastas[i], columns=colnames)
        df.to_csv(path, index=False)
        print("Save preprocessed file to %s" % path)
    return paths
# < 40% identity to novel, others randomly select 10% as test,
# remained select 10% as val-train, last 90% as tra-train
def SplitSpecie2TrainValTestNovel(specie="escherichia coli", thr=0, cdhit_flag=False):
    spe = GetOriSpecieInfo(specie)
    colnames, abb = spe.colnames, spe.specie_abb
    fastas = spe.fastas.to_numpy().tolist()
    records, short_fastas, path = fastas2Records(fastas, thr)
    fastas = {}
    if cdhit_flag:
        msg, out_folder, novel_path, test_cdhit_path, train_cdhit_path = executeCdhitAndSplitDataset(path, del_novel=False)
        test_cdhit_fastas = readfastasFromFastaPath(test_cdhit_path)
        train_cdhit_fastas = readfastasFromFastaPath(train_cdhit_path)
        fastas["test_cdhit"] = test_cdhit_fastas
        fastas["train_cdhit"] = train_cdhit_fastas
        fasta_names = ["train_cdhit", "test_cdhit"]
    else:
        msg, out_folder, novel_path, all_path = executeCdhitAndSplitDataset(path, del_novel=True)
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
    print("Save preprocessed file to %s" % path)
    for i in fasta_names:
        path = os.path.join(mrg_dir, "%s_%s" % (i, name))
        paths[i] = path
        df = pd.DataFrame(fastas[i], columns=colnames)
        df.to_csv(path, index=False)
        print("Save preprocessed file to %s" % path)

    if cdhit_flag:
        tra_cdhit_path, val_cdhit_path = splitValidationFromTrain(paths['train_cdhit'])
        paths["%s-train" % abb] = train_cdhit_path
        paths["%s-test" % abb] = test_cdhit_path
        paths["%s-tra-train" % abb] = tra_cdhit_path
        paths["%s-val-train" % abb] = val_cdhit_path
    else:
        print("saved %s fastas to: \n\t%s" % (specie, paths['delNovel']))
        train_path, test_path = splitTestFromAll(paths['delNovel'])
        tra_path, val_path = splitValidationFromTrain(train_path)
        paths["%s-train" % abb] = train_path
        paths["%s-test" % abb] = test_path
        paths["%s-tra-train" % abb] = tra_path
        paths["%s-val-train" % abb] = val_path
    return paths
