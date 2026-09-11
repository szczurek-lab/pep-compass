# from tools_GeneFt import *
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os
from glob import glob
import pandas as pd
import numpy as np
# from tools.base import GetFolder
from tools.base import *
from tools.cdhit import *
host_specie = "escherichia coli"
target_species = ["staphylococcus aureus", "enterococcus faecium", "streptococcus pneumoniae", "bacillus subtilis"]
vip_species = ["escherichia coli", "staphylococcus aureus"]
unnatural_amino_acids = ["B", "J", "O", "U", "Z", "X"]
largest_MIC = 10000; def_bias = 3.8; def_scale = 1/6;
def_lrParas = {"lr": 5e-4 ,"decay_rate": 0.92, "decay_steps": 25, "patience": 15, "monitor": "loss"}
def_hyperParaDict = {"layer_num": 1, "dropOutRate": 0.4, "filter_num": 32, "epoch": 200}
def_fts = \
['type8raac9glmd3lambda-correlation', 'type8raac7glmd3lambda-correlation', 'QSOrder_lmd4', 'QSOrder_lmd3', 'QSOrder_lmd2',
 'QSOrder_lmd1', 'QSOrder_lmd0', 'type5raac15glmd4lambda-correlation', 'type7raac10glmd3lambda-correlation',
 'type5raac8glmd2lambda-correlation', 'type3Braac9glmd3lambda-correlation', 'type2raac15glmd4lambda-correlation',
 'type2raac8glmd2lambda-correlation', 'type8raac14glmd1lambda-correlation']

# largest_MIC = 10000;  EC pMIC range: -3.7781512503836434 ~ 2.1630917510364918
# largest_MIC = 10; def_bias = 1; def_scale = 1/3.2; #  EC pMIC range: -0.9997855462266436 ~ 2.1630917510364918

def getOriSepcieLocation(specie_name):
    fld = GetOriFolder()
    sepcie_path = os.path.join(fld.data_dir,  "specie_%s.csv" % specie_name.replace(' ', '-'))
    return sepcie_path

def getSepcieLocation(specie_name):
    from tools.base import GetFolder, abbreviation
    fld = GetFolder()
    abb = abbreviation(specie_name)
    sepcie_path = os.path.join(fld.data_dir,  "%s.csv" % abb)
    return sepcie_path, abb

def controlSeqsLengthRange(infos, shorest_len=5, largerest_len=60):
    infos = infos.copy()
    seqs = infos["SEQUENCE"].to_list()
    for seq in seqs:
        l = len(seq)
        if l < shorest_len or l > largerest_len:
            infos = infos[infos["SEQUENCE"] != seq]
    return infos

def controlMICvalue(infos, largest_MIC=largest_MIC):
    infos = infos.copy()
    infos = infos[infos['TARGET ACTIVITY - CONCENTRATION - PROCED'] < largest_MIC]
    return infos

class GetOriSpecieInfo():
    def __init__(self, specie_name="escherichia coli", control_seqLens=True, control_MIC=True, largest_MIC=largest_MIC):
        self.specie_path = getOriSepcieLocation(specie_name)
        self.specie_abb = abbreviation(specie_name)
        specie_info = pd.read_csv(self.specie_path)
        # only extract seqs whose N and C terminus are np.nan
        ind = pd.isna(specie_info["N TERMINUS"]) & pd.isna(specie_info["C TERMINUS"])
        specie_info = specie_info[ind]
        if control_seqLens:
            specie_info = controlSeqsLengthRange(specie_info, shorest_len=5, largerest_len=60)
        if control_MIC:
            specie_info = controlMICvalue(specie_info, largest_MIC=largest_MIC)
        self.specie_info = specie_info
        self.pMIC_name = '%s_pMIC' % self.specie_abb
        self.MIC_name = '%s_MIC' % self.specie_abb
        self.pMIC = -np.log10(self.specie_info['TARGET ACTIVITY - CONCENTRATION - PROCED'])
        self.specie_info[self.pMIC_name] = self.pMIC
        self.mic = self.specie_info['TARGET ACTIVITY - CONCENTRATION - PROCED'].to_list()
        self.specie_info[self.MIC_name] = self.mic
        fastas = self.specie_info[["ID", "SEQUENCE", self.MIC_name, self.pMIC_name]]
        self.fastas = fastas
        self.colnames = fastas.columns.to_list()
        printDataInfo("%s Info: " % specie_name, self.specie_path, self.specie_info)

def getPathsFormOnlySpeciePathRepVal(specie_name="escherichia coli", threshold=largest_MIC, del_novel=False, cdhit_flag=False, rep_tuple=0):
    paths = {}
    fld = GetFolder()
    abb = abbreviation(specie_name)
    if del_novel:
        file_name = "delNovel_%s.csv" % abb
    else:
        file_name = "%s.csv" % abb
    data_dir = os.path.join(fld.data_dir, "MIC%d" % threshold)
    val_dir = os.path.join(data_dir, "test%d" % rep_tuple[0], "val%d" % rep_tuple[1])
    test_dir = os.path.join(data_dir, "test%d" % rep_tuple[0])
    file_path = os.path.join(data_dir, file_name)
    train_name, test_name = "train-%s" % file_name, "test-%s" % file_name
    if cdhit_flag:
        file_name = "%s.csv" % abb
        train_name, test_name = "train_cdhit_%s" % file_name, "test_cdhit_%s" % file_name
    train_path = os.path.join(test_dir, train_name)
    test_path = os.path.join(test_dir, test_name)
    tra_path = os.path.join(val_dir, "tra-%s" % train_name)
    val_path = os.path.join(val_dir, "val-%s" % train_name)
    paths[abb] = file_path
    paths["train"] = train_path
    paths["test"] = test_path
    paths["tra-train"] = tra_path
    paths["val-train"] = val_path
    return paths, abb


def getPathsFormOnlySpeciePath(specie_name="escherichia coli", threshold=largest_MIC, del_novel=False, cdhit_flag=False, rep=0):
    from tools.base import GetFolder, abbreviation
    paths = {}
    fld = GetFolder()
    abb = abbreviation(specie_name)
    if del_novel:
        file_name = "delNovel_%s.csv" % abb
    else:
        file_name = "%s.csv" % abb
    data_dir = os.path.join(fld.data_dir, "MIC%d" % threshold)
    val_dir = os.path.join(data_dir, str(rep))
    test_dir = val_dir
    file_path = os.path.join(data_dir, file_name)
    train_name, test_name = "train-%s" % file_name, "test-%s" % file_name
    if cdhit_flag:
        file_name = "%s.csv" % abb
        train_name, test_name = "train_cdhit_%s" % file_name, "test_cdhit_%s" % file_name
    train_path = os.path.join(test_dir, train_name)
    test_path = os.path.join(test_dir, test_name)
    tra_path = os.path.join(val_dir, "tra-%s" % train_name)
    val_path = os.path.join(val_dir, "val-%s" % train_name)
    paths[abb] = file_path
    paths["train"] = train_path
    paths["test"] = test_path
    paths["tra-train"] = tra_path
    paths["val-train"] = val_path
    return paths, abb

class GetSpecieInfo():
    def __init__(self, specie_name="escherichia coli", del_novel=False, cdhit_flag=False, rep=0):
        from tools.base import printDataInfo
        self.specie_path, self.abb = getSepcieLocation(specie_name)
        paths, abb = getPathsFormOnlySpeciePath(specie_name, del_novel=del_novel, cdhit_flag=cdhit_flag, rep=rep)
        self.abb = abb
        self.specie_path, self.train_path = paths[abb], paths["train"]
        self.test_path, self.tra_train_path = paths["test"], paths["tra-train"]
        self.val_train_path = paths["val-train"]
        specie_info = pd.read_csv(self.specie_path)
        self.specie_info = specie_info
        self.fastas = self.specie_info
        printDataInfo("%s Info: " % specie_name, self.specie_path, self.specie_info)
        self.train_fastas = pd.read_csv(self.train_path)
        printDataInfo("%s train Info: " % specie_name, self.train_path, self.train_fastas)
        self.test_fastas = pd.read_csv(self.test_path)
        printDataInfo("%s test Info: " % specie_name, self.test_path, self.test_fastas)
        self.tra_train_fastas = pd.read_csv(self.tra_train_path)
        printDataInfo("%s tra of train Info: " % specie_name, self.tra_train_path, self.tra_train_fastas)
        self.val_train_fastas = pd.read_csv(self.val_train_path)
        printDataInfo("%s val of train Info: " % specie_name, self.val_train_path, self.val_train_fastas)


def writeFastasfromInfo(info, columns, path):
    fastas = info[columns]
    fastas.to_csv(path, index=False)
    return

def splitRndTrainTestFromWholePath(file_path, x=20):
    spl_path = file_path.split("/")
    train_path = os.path.join("/".join(spl_path[:-1]), "train-%dPercent-%s" % (100-x, spl_path[-1]))
    test_path = os.path.join("/".join(spl_path[:-1]), "test-%dPercent-%s" % (x, spl_path[-1]))
    df = pd.read_csv(file_path)
    test = df.sample(frac=x/100, replace=False).copy()
    train = handleInfos(df, test).infos1DelInfos2()
    test.to_csv(test_path)
    print("whole sample: ", len(df), "\ntrain sample: ", len(train), "\ntest sample: ", len(test))
    print("saved %d percent samples of whole set (%s): \n\t%s" % (x, spl_path[-1], test_path))
    train.to_csv(train_path)
    print("saved %d percent samples of whole set (%s): \n\t%s" % (100-x, spl_path[-1], train_path))
    return train_path, test_path

def splitValidationFromTrain(file_path, x=10, rep=0, valDoNotRep=True):
    spl_path = file_path.split("/")
    if not valDoNotRep:
        rep_dir = os.path.join("/".join(spl_path[:-1]), "val%d" % rep)
    else:
        rep_dir = os.path.join("/".join(spl_path[:-1]))
    createFolder(rep_dir)
    tra_path = os.path.join(rep_dir, "tra-%s" % (spl_path[-1]))
    val_path = os.path.join(rep_dir, "val-%s" % (spl_path[-1]))
    df = pd.read_csv(file_path)
    val = df.sample(frac=x/100, replace=False).copy()
    train = handleInfos(df, val).infos1DelInfos2()
    val.to_csv(val_path)
    print("train sample: ", len(df), "\ntra train sample: ", len(train), "\nval train sample: ", len(val))
    print("saved %d percent samples of trainset (%s): \n\t%s" % (x, spl_path[-1], val_path))
    train.to_csv(tra_path)
    print("saved %d percent samples of trainset (%s): \n\t%s" % (100-x, spl_path[-1], tra_path))
    return tra_path, val_path

def splitTestFromAll(file_path, x=10, rep=0, valDoNotRep=True):
    spl_path = file_path.split("/")
    if valDoNotRep:
        rep_dir = os.path.join("/".join(spl_path[:-1]), str(rep))
    else:
        rep_dir = os.path.join("/".join(spl_path[:-1]), "test%d" % rep)
    createFolder(rep_dir)
    train_path = os.path.join(rep_dir, "train-%s" % (spl_path[-1]))
    test_path = os.path.join(rep_dir, "test-%s" % (spl_path[-1]))
    df = pd.read_csv(file_path)
    test = df.sample(frac=x/100, replace=False).copy()
    train = handleInfos(df, test).infos1DelInfos2()
    test.to_csv(test_path)
    print("all sample: ", len(df), "\ntrain sample: ", len(train), "\ntest sample: ", len(test))
    print("saved %d percent samples of trainset (%s): \n\t%s" % (x, spl_path[-1], test_path))
    train.to_csv(train_path)
    print("saved %d percent samples of trainset (%s): \n\t%s" % (100-x, spl_path[-1], train_path))
    return train_path, test_path

def geneAlltest(host_specie="escherichia coli", target_specie="staphylococcus aureus", onlyMrg=True, cdhit=True):
    mrg = GetMergeInfo(host_specie=host_specie, target_specie=target_specie, onlyMrg=onlyMrg, cdhit=cdhit)
    df = pd.concat((mrg.test_fastas, mrg.novel_fastas))
    path = mrg.mrg_path
    spl_path = path.split("/")
    path = os.path.join("/".join(spl_path[:-1]), "all_test-%s" % (spl_path[-1]))
    df.to_csv(path)
    print("saved all test of %s merge %s: \n\t%s" % (host_specie, target_specie, path))
    return df

def PreProcessedSpecies(host_specie="escherichia coli", target_specie="staphylococcus aureus"):
    """
    control the sequence length in [5,60], MIC value < 1000 μM;
    split the merge data to train/ test/ novel datasets
    :param host_specie:
    :param target_specie:
    :return:
    """
    # read original host specie info and save fastas to data folder
    paths = {}
    hst = GetOriSpecieInfo(host_specie)
    hst_path, hst_abb = getSpeciePath(host_specie)
    hst.fastas.to_csv(hst_path, index=False)
    paths[hst_abb] = hst_path
    print("saved %s fastas to: \n\t%s" % (host_specie, hst_path))
    # read target host specie info and save fastas to data folder
    tgt = GetOriSpecieInfo(target_specie)
    tgt_path, tgt_abb = getSpeciePath(target_specie)
    tgt.fastas.to_csv(tgt_path, index=False)
    paths[tgt_abb] = tgt_path
    print("saved %s fastas to: \n\t%s" % (target_specie, tgt_path))
    fld = GetMdlFolder(host_specie=host_specie, target_specie=target_specie)
    # save intersection of hst and tgt
    colnames = getMergeFastaColnames(host_specie=host_specie, target_specie=target_specie).colnames
    mrg_info, pmicr_name = handleInfos(hst.fastas, tgt.fastas).infos1IntersectInfos2(hst_abb, tgt_abb)
    mrg_info = mrg_info[colnames].copy()
    mrg_path, hst_abb, tgt_abb, mrg_abb = getMergePath(host_specie=host_specie, target_specie=target_specie)
    mrg_info.to_csv(mrg_path, index=False)
    # mrg_abb = "%s_mrg_%s" % (tgt_abb, hst_abb)
    paths[mrg_abb] = mrg_path
    print("saved %s merged %s to: \n\t%s" % (host_specie, target_specie, mrg_path))
    # split merge to train test novel
    paths[mrg_abb] = SplitMerge2TrainTestNovel(host_specie=host_specie, target_specie=target_specie)
    # since we have generated the novel and test dataset, then we can use GetMergeInfo() now.
    # save all-test of hst and tgt
    all_test = geneAlltest(host_specie=host_specie, target_specie=target_specie, onlyMrg=True, cdhit=True)
    # save del merge all-test of hst and tgt
    mrg = GetMergeInfo(host_specie=host_specie, target_specie=target_specie, onlyMrg=True, cdhit=True)
    hst_del_mrg = handleInfos(hst.fastas, all_test).infos1DelInfos2()
    hst_name, train_name = os.path.basename(hst_path), os.path.basename(mrg.mrg_path)
    hst_del_mrg_name = "%s-del-all_test-%s" % (hst_name.split(".csv")[0], train_name)
    hst_del_mrg_path = os.path.join(fld.data_dir, hst_del_mrg_name)
    hst_del_mrg.to_csv(hst_del_mrg_path, index=False)
    paths["%s_del_all-test_%s" % (hst_abb, mrg_abb)] = hst_del_mrg_path
    print("saved %s delete %s to: \n\t%s" % (host_specie, mrg_abb, hst_del_mrg_path))
    tgt_del_mrg = handleInfos(tgt.fastas, all_test).infos1DelInfos2()
    tgt_name = os.path.basename(tgt_path)
    tgt_del_mrg_name = "%s-del-all_test-%s" % (tgt_name.split(".csv")[0], train_name)
    tgt_del_mrg_path = os.path.join(fld.data_dir, tgt_del_mrg_name)
    tgt_del_mrg.to_csv(tgt_del_mrg_path, index=False)
    paths["%s_del_all-test_%s" % (tgt_abb, mrg_abb)] = tgt_del_mrg_path
    print("saved %s delete %s to: \n\t%s" % (host_specie, mrg_abb, tgt_del_mrg_path))
    # split all the train sets to validation-train (10% train) and train-train (90% train) subsets
    for path in [mrg.train_path, hst_del_mrg_path, tgt_del_mrg_path]:
        tra_path, val_path = splitValidationFromTrain(path)
    return paths

def PreProcessedSpeciesWithoutCDhit(host_specie="escherichia coli", target_specie="staphylococcus aureus", x=20, threshold=largest_MIC):
    """
    control the sequence length in [5,60], MIC value < 1000 μM;
    split the merge data to train/ test/ novel datasets
    :param host_specie:
    :param target_specie:
    x is the percent of test ratio compare to whole
    :return:
    """
    # read original host specie info and save fastas to data folder
    paths = {}
    hst = GetOriSpecieInfo(host_specie)
    hst_path, hst_abb = getSpeciePath(host_specie)
    hst.fastas.to_csv(hst_path, index=False)
    paths[hst_abb] = hst_path
    print("saved %s fastas to: \n\t%s" % (host_specie, hst_path))
    # read target host specie info and save fastas to data folder
    tgt = GetOriSpecieInfo(target_specie)
    tgt_path, tgt_abb = getSpeciePath(target_specie)
    tgt.fastas.to_csv(tgt_path, index=False)
    paths[tgt_abb] = tgt_path
    print("saved %s fastas to: \n\t%s" % (target_specie, tgt_path))
    fld = GetMdlFolder(host_specie=host_specie, target_specie=target_specie)
    # save intersection of hst and tgt
    colnames = getMergeFastaColnames(host_specie=host_specie, target_specie=target_specie).colnames
    mrg_info, pmicr_name = handleInfos(hst.fastas, tgt.fastas).infos1IntersectInfos2(hst_abb, tgt_abb)
    mrg_info = mrg_info[colnames].copy()
    mrg_path, hst_abb, tgt_abb, mrg_abb = getMergePath(host_specie=host_specie, target_specie=target_specie)
    mrg_info.to_csv(mrg_path, index=False)
    # mrg_abb = "%s_mrg_%s" % (tgt_abb, hst_abb)
    paths[mrg_abb] = mrg_path
    print("saved %s merged %s to: \n\t%s" % (host_specie, target_specie, mrg_path))
    # split merge to train and test randomly, and keep ratio to 8 : 2
    mrg_train_path, mrg_test_path = splitRndTrainTestFromWholePath(file_path=mrg_path, x=x)
    # since we have generated the novel and test dataset, then we can use GetMergeInfo() now.
    # save del merge test of hst and tgt
    mrg = GetMergeInfo(host_specie=host_specie, target_specie=target_specie, onlyMrg=True, cdhit=False, x=x)
    hst_del_mrg_test = handleInfos(hst.fastas, mrg.test_fastas).infos1DelInfos2()
    hst_name, test_name = os.path.basename(hst_path), os.path.basename(mrg.test_path)
    hst_del_mrg_name = "%s-del-%s" % (hst_name.split(".csv")[0], test_name)
    hst_del_mrg_test_path = os.path.join(fld.data_dir, "MIC%s" % threshold, hst_del_mrg_name)
    hst_del_mrg_test.to_csv(hst_del_mrg_test_path, index=False)
    paths["%s_del_%s" % (hst_abb, mrg_abb)] = hst_del_mrg_test_path
    print("saved %s delete test of %s to: \n\t%s" % (host_specie, mrg_abb, hst_del_mrg_test_path))
    tgt_del_mrg_test = handleInfos(tgt.fastas, mrg.test_fastas).infos1DelInfos2()
    tgt_name = os.path.basename(tgt_path)
    tgt_del_mrg_test_name = "%s-del-%s" % (tgt_name.split(".csv")[0], test_name)
    tgt_del_mrg_test_path = os.path.join(fld.data_dir, "MIC%s" % threshold, tgt_del_mrg_test_name)
    tgt_del_mrg_test.to_csv(tgt_del_mrg_test_path, index=False)
    paths["%s_del_%s" % (tgt_abb, mrg_abb)] = tgt_del_mrg_test_path
    print("saved %s delete test of %s to: \n\t%s" % (host_specie, mrg_abb, tgt_del_mrg_test_path))
    for path in [hst_del_mrg_test_path, tgt_del_mrg_test_path]:
        tra_path, val_path = splitValidationFromTrain(path)
    return paths


def getSingleSpeciePath(specie="escherichia coli"):
    """
    control the sequence length in [5,60], MIC value < 1000 μM;
    split the merge data to train/ test/ novel datasets
    :param specie: specie name
    :return:
    """
    path, abb = getSpeciePath(specie)
    return

def PreProcessedSingleSpecies(specie="escherichia coli", x=10, reps=3, valDoNotRep=True):
    """
    control the sequence length in [5,60], MIC value < 1000 μM;
    split the single data to train (tra-train/val-train) / test datasets ratio = 9:1 randomly
    :param specie: specie name
    :return:
    """
    # read original host specie info and save fastas to data folder
    paths = {}
    spe = GetOriSpecieInfo(specie)
    path, abb = getSpeciePath(specie)
    dir = os.path.dirname(path)
    createFolder(dir)
    spe.fastas.to_csv(path, index=False)
    paths[abb] = path
    print("saved %s fastas to: \n\t%s" % (specie, path))
    if valDoNotRep:
        for i in range(reps):
            train_path, test_path = splitTestFromAll(path, x=x, rep=i, valDoNotRep=valDoNotRep)
            tra_path, val_path = splitValidationFromTrain(train_path, x=x, rep=i, valDoNotRep=valDoNotRep)
            paths["%s-tra-train%d" % (abb, i)] = tra_path
            paths["%s-val-train%d" % (abb, i)] = val_path
            paths["%s-train%d" % (abb, i)] = train_path
            paths["%s-test%d" % (abb, i)] = test_path
    else:
        for i in range(reps):
            train_path, test_path = splitTestFromAll(path, x=x, rep=i, valDoNotRep=valDoNotRep)
            for j in range(reps):
                tra_path, val_path = splitValidationFromTrain(train_path, x=x, rep=j, valDoNotRep=valDoNotRep)
                paths["%s-tra%d-train%d" % (abb, j, i)] = tra_path
                paths["%s-val%d-train%d" % (abb, j, i)] = val_path
            paths["%s-train%d" % (abb, i)] = train_path
            paths["%s-test%d" % (abb, i)] = test_path
    return paths

def PreProcessedSingleSpeciesDelNovel(specie="escherichia coli", cdhit_flag=False):
    """
    control the sequence length in [5,60], MIC value < 1000 μM;
    split the single-novel data to train (tra-train, val-train), test  datasets randomly 9:1
    :param specie: specie name
    :return:
    """
    # read original host specie info and save fastas to data folder
    paths = SplitSpecie2TrainValTestNovel(specie=specie, thr=0, cdhit_flag=cdhit_flag)
    return paths

def preProcessAllData():
    host_specie = "escherichia coli"
    target_species = ["staphylococcus aureus", "enterococcus faecium", "streptococcus pneumoniae", "bacillus subtilis"]
    # do preprocessing uncomment next two lines
    for target_specie in target_species:
        paths = PreProcessedSpecies(host_specie=host_specie, target_specie=target_specie)

    # print data informations:
    print("\n\n*****************Starting print single specie info*****************")
    for specie_name in ["escherichia coli", "staphylococcus aureus", "enterococcus faecium", "streptococcus pneumoniae", "bacillus subtilis"]:
        spe = GetSpecieInfo(specie_name)
    print("\n\n*****************Starting print merge info*****************")
    for target_specie in target_species:
        mrg = GetMergeInfo(host_specie=host_specie, target_specie=target_specie)
    print("\n\n*****************Starting print delete merge info*****************")
    for target_specie in target_species:
        dlt = GetDelInfo(host_specie=host_specie, target_specie=target_specie)
    return


if __name__ == "__main__":
    specie = "escherichia coli"
    # paths = PreProcessedSingleSpecies(specie=specie)

    # host_specie = "escherichia coli"
    # target_species = ["staphylococcus aureus", "enterococcus faecium", "streptococcus pneumoniae", "bacillus subtilis"]
    # # do preprocessing uncomment next two lines
    # for target_specie in target_species:
    #     paths = PreProcessedSpecies(host_specie=host_specie, target_specie=target_specie)
    #
    # # print data informations:
    # print("\n\n*****************Starting print single specie info*****************")
    # for specie_name in ["escherichia coli", "staphylococcus aureus", "enterococcus faecium", "streptococcus pneumoniae", "bacillus subtilis"]:
    #     spe = GetSpecieInfo(specie_name)
    # print("\n\n*****************Starting print merge info*****************")
    # for target_specie in target_species:
    #     mrg = GetMergeInfo(host_specie=host_specie, target_specie=target_specie)
    # print("\n\n*****************Starting print delete merge info*****************")
    # for target_specie in target_species:
    #     dlt = GetDelInfo(host_specie=host_specie, target_specie=target_specie)

    # species = ["staphylococcus aureus", "enterococcus faecium", "streptococcus pneumoniae"]
    # spes = []
    # for i in range(len(species)):
    #     specie_name = species[i]
    #     spe = GetOriSpecieInfo(specie_name)
    #     spes.append(spe)
    # infos = handleInfos(spes[0].fastas, spes[1].fastas).infos1UnionInfos2()
    # infos = handleInfos(infos, spes[2].fastas).infos1UnionInfos2()