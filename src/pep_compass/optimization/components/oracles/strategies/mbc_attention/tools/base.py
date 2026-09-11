import os
from glob import glob
import numpy as np
import pandas as pd
from pycaret.utils.generic import check_metric
from pycaret.regression import add_metric, get_metrics, remove_metric
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
# import random
# import torch
from tools.preProcessed import largest_MIC, splitValidationFromTrain
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
metrics = ['R2', 'PCC', 'CCC', 'MAE', 'MSE', 'RMSE', 'RMSLE']
host_specie = "escherichia coli"
target_specie = ["staphylococcus aureus", "enterococcus faecium", "streptococcus pneumoniae", "bacillus subtilis"]
vip_species = ["escherichia coli", "staphylococcus aureus"]
unnatural_amino_acids = ["B", "J", "O", "U", "Z", "X"]

def readFastaYan(fasta_file):
    seqs = []
    lens = []
    names = []
    records =[]
    for record in SeqIO.parse(fasta_file, "fasta"):
        # read Sequences
        seqs.append(str(record.seq))
        # get sequence length
        lens.append(len(record.seq))
        # get sequence name record.id = "P61542|ori"
        name = record.id.split("|")[0]
        names.append(name)
        record.id = name
        records.append(record)
    return names, seqs, lens, records

def geneFastasFromFastaFile(fasta_file):
    seqs = []
    lens = []
    names = []
    records =[]
    names, seqs, lens, records = readFastaYan(fasta_file)
    fastas_file = fasta_file.replace(".fasta", ".csv")
    fastas = pd.DataFrame({"ID": names, "SEQUENCE": seqs})
    fastas.to_csv(fastas_file, header=True, index=False)
    print("fasta file was Converted to fastas which is a csv file with ID and SEQUENCE columns: \n\t", fastas_file)
    return fastas, fastas_file

class handleInfos():
    def __init__(self, infos1, infos2):
        self.infos1, self.infos2 = infos1, infos2

    def infos1DelInfos2(self):
        infos1, infos2 = self.infos1, self.infos2
        seqs1 = infos1["SEQUENCE"].to_list()
        seqs2 = infos2["SEQUENCE"].to_list()
        infos = infos1.copy()
        for seq in seqs2:
            if seq in seqs1:
                infos = infos[infos["SEQUENCE"] != seq]
        return infos

    def infos1IntersectInfos2(self, hst_abb, tgt_abb):
        infos1, infos2 = self.infos1, self.infos2
        infos = pd.merge(infos1, infos2, how='inner', on=['ID', 'SEQUENCE'])
        # calculate the pMICR of target and
        info = infos.copy()
        pMIC_name1, pMIC_name2 = "%s_pMIC" % hst_abb, "%s_pMIC" % tgt_abb
        MIC_name1, MIC_name2 = "%s_MIC" % hst_abb, "%s_MIC" % tgt_abb
        hst, tgt = info[MIC_name1], info[MIC_name2]
        mic_ratio = - np.log10(tgt / hst)
        mic_ratio = mic_ratio.to_numpy().tolist()
        pMICR_name = "%s_pMICR_%s" % (tgt_abb, hst_abb)
        info[pMICR_name] = mic_ratio
        return info, pMICR_name

    def infos1UnionInfos2(self):
        infos1, infos2 = self.infos1, self.infos2
        seqs1 = infos1["SEQUENCE"].to_list()
        seqs2 = infos2["SEQUENCE"].to_list()
        infos = pd.concat((infos1, infos2))
        infos = infos.drop_duplicates(subset=['SEQUENCE']).copy()
        return infos

# def seed_everything(seed: int):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
#seed_everything(seed=0)

def abbreviation(string):
    words = string.split(" ")
    abb = ""
    for word in words:
        first_char = word[0]
        abb = abb + first_char.capitalize()
    return abb

def getRootDir():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = "/".join(root_dir.split("/")[:-1])
    return root_dir

def createFolder(folder, slient=True):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("created folder: \n\t %s" % folder)
    else:
        if not slient:
            print("%s existed." % folder)
    return


import dill
def write_pkl(obj, file_path):
    with open(file_path, 'wb') as f:
        dill.dump(obj, f)
    return

def read_pkl(file_path):
    with open(file_path, "rb") as f:
        final_rs = dill.load(f)
    return final_rs

def calRegMetrics(ori, pre, metric_names=['R2', 'PCC', 'CCC', 'MAE', 'MSE', 'RMSE', 'RMSLE']):
    metrics = {}
    # replaceMetrics()
    for m in metric_names:
        metric = check_metric(ori, pre, m)
        metrics[m] = metric
    metrics = pd.DataFrame(metrics, index=[0], columns=metric_names)
    return metrics

def binary(num, digits):
    onezero = f'{int(num*2**digits):b}'
    # this is real binary value with:
    # return f'{onezero[:-digits]}.{onezero[-digits:]}'
    # output an int array
    return [int(x) for x in onezero]

def dec2binFloat(x):
    x -= int(x)
    bins = []
    while x:
        x *= 2
        bins.append(1 if x>=1. else 0)
        x -= int(x)
    return bins

def dec2binInt(x):
    bin_x = bin(x)
    bins = [int(i) for i in bin_x[2:]]
    return bins

def bin2decFloat(b):
    d = 0
    for i, x in enumerate(b):
        x = int(x)
        d += 2**(-i-1)*x
    return d

def bin2decInt(b):
    d = 0
    for i, x in enumerate(b):
        d += 2**(i+1)*x
    return d

def dec2bin(x, digits=15):
    if x < 0:
        print("warning: the MIC/pMICR is %.5f, but it should be greater than 0.\n and 0 will be returned" % x)
        return [0] * digits
    else:
        bins = []
        s_int, s_float = str(x).split(".")
        x_int = int(s_int)
        bins_int = dec2binInt(x_int)
        zeros = [0] * (digits - len(bins_int))
        bins.extend(zeros)
        bins.extend(bins_int)
        x_float = float("0." + s_float)
        bins_float = dec2binFloat(x_float)
        zeros = [0] * (digits - len(bins_float))
        bins.extend(bins_float)
        bins.extend(zeros)
    return bins

def binaryArrayForMulitple(digits=15):
    p_int = [2 ** (digits-1-i) for i in range(digits)]
    p_float = [2 ** (-i-1) for i in range(digits)]
    ps = []
    ps.extend(p_int)
    ps.extend(p_float)
    binary_elements = np.array(ps)
    return binary_elements

binary_elements = binaryArrayForMulitple(digits=15)
def bin2dec(bs, digits=15):
    binary_elements = [2 ** i for i in range(digits-1,-digits -1,-1)]
    xs = np.multiply(binary_elements, bs)
    x = np.sum(xs)
    return x

def MSE(ori, pre):
    return np.square(np.subtract(ori, pre)).mean()

def RMSE(ori, pre):
    mse = np.square(np.subtract(ori, pre)).mean()
    return np.sqrt(mse)

def PCCC(output, target):
    eps = 1e-6
    x_ave, y_ave = np.mean(target), np.mean(output)
    vx, vy = target - x_ave, output - y_ave
    x_std, y_std = np.std(target), np.std(output)
    x_var, y_var = np.var(target), np.var(output)
    pcc_value = np.sum(vx*vy) / (np.sqrt(np.sum(vx ** 2) + eps) * np.sqrt(np.sum(vy ** 2) + eps))
    ccc_value = (2*pcc_value*x_std*y_std) / (x_var + y_var + (x_ave - y_ave)**2)
    return pcc_value, ccc_value

def PCC(ori, pre):
    pcc, ccc = PCCC(pre, ori)
    return pcc

def CCC(ori, pre):
    pcc, ccc = PCCC(pre, ori)
    return ccc

def RMSLE(y_true, y_pred):
    n = len(y_true)
    #torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    #e = 10**-16
    msle = np.mean([(np.log((y_pred[i] + 1)/(y_true[i] + 1))) ** 2.0 for i in range(n)])
    return np.sqrt(msle)
    # return np.sqrt(mean_squared_log_error(ori, pre))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calMetrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = MSE(y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pcc, ccc = PCCC(y_pred, y_true)
    # rmsle = RMSLE(y_true, y_pred)
    l = len(y_true)
    metrics = pd.DataFrame(data=[[r2, pcc, ccc, mae, mse, rmse, l]],
                           columns=['R2', 'PCC', 'CCC', 'MAE', 'MSE', 'RMSE', 'N0.'])
    return metrics

def replaceMetrics():
    # all_metrics = get_metrics(reset=True)
    remove_metric("R2")
    add_metric("R2", "R2", r2_score)
    # remove_metric("PCC")
    add_metric("PCC", "PCC", PCC)
    # remove_metric("CCC")
    add_metric("CCC", "CCC", CCC)
    remove_metric("MAE")
    add_metric("MAE", "MAE", mean_absolute_error, greater_is_better=False)
    remove_metric("MSE")
    add_metric("MSE", "MSE", MSE, greater_is_better=False)
    remove_metric("RMSE")
    add_metric("RMSE", "RMSE", RMSE, greater_is_better=False)
    remove_metric("RMSLE")
    add_metric("RMSLE", "RMSLE", mean_squared_log_error, greater_is_better=False)
    remove_metric("MAPE")

    # add_metric("MAPE", "MAPE", mean_absolute_percentage_error)
    # R: pearson correlation coefficient
    return

def getUnexistedName(file_path, file_type=".xlsx"):
    # print("file_path: ", file_path)
    file_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    if not(os.path.isfile(file_path) or os.path.isdir(file_path)):
        new_file = file_name
    else:
        name = file_name.split(file_type)[0]
        regex_str = name[:-1] + "*" + file_type
        regex_str = os.path.join(dir_name, regex_str)
        # print("regex_str: ", regex_str)
        mylist = [f for f in glob(regex_str)]
        s = sorted(mylist)
        s = [os.path.basename(i) for i in s]
        s.remove(file_name)
        if len(s) > 0:
            nums = []
            rms = []
            for n in s:
                num_name = n.split(file_type)[0]
                suffix = num_name.split(file_name.split(file_type)[0])[-1]
                str_i = [str(i) for i in range(10)]
                str_num = suffix.split("-")[-1]
                a = [i in str_i for i in str_num]
                if len(str_num) == sum(a):
                    num = int(str_num)
                else:
                    rms.append(n)
                    continue
                if "num" in locals():
                    nums.append(num)
                # nums.append(num)
            for rm in rms:
                s.remove(rm)
            if len(nums) > 0:
                max_n = sorted(nums)[-1]
        if len(s) == 0:
            new_file = name + "-0" + file_type
        else:
            new_file = name + "-" + str(max_n+1) + file_type
    new_path = os.path.join(dir_name, new_file)
    return new_path

def getExistedLargestName(file_path, file_type=".xlsx"):
    file_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    if not(os.path.isfile(file_path)):
        new_file = file_name
        print("%s unexisted." % file_path)
    else:
        name = file_name.split(file_type)[0]
        regex_str = name[:-1] + "*" + file_type
        regex_str = os.path.join(dir_name,regex_str)
        mylist = [f for f in glob(regex_str)]
        s = sorted(mylist)
        s = [os.path.basename(i) for i in s]
        s.remove(file_name)
        if len(s) > 0:
            nums = []
            rms = []
            for n in s:
                num_name = n.split(file_type)[0]
                suffix = num_name.split(file_name.split(file_type)[0])[-1]
                str_i = [str(i) for i in range(10)]
                str_num = suffix.split("-")[-1]
                a = [i in str_i for i in str_num]
                if len(str_num) == sum(a):
                    num = int(str_num)
                else:
                    rms.append(n)
                    continue
                if "num" in locals():
                    nums.append(num)
            for rm in rms:
                s.remove(rm)
            if len(nums) > 0:
                max_n = sorted(nums)[-1]
        if len(s) == 0:
            new_file = file_name
        else:
            new_file = name + "-" + str(max_n) + file_type
    new_path = os.path.join(dir_name, new_file)
    return new_path

def printDataInfo(first_line, path, data, colPrint=False):
    print(first_line)
    print("\tPath: %s" % path)
    print("\tSample No.: %d" % len(data))
    if colPrint:
        print("\tFeature No.: %d" % data.shape[-1])
    return

class GetOriFolder():
    def __init__(self):
        self.root_dir = getRootDir()
        self.data_dir = os.path.join(self.root_dir, 'split_data')
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        # self.mrg_dir = os.path.join(self.data_dir, 'merge')
        # if not os.path.isdir(self.mrg_dir):
        #     os.mkdir(self.mrg_dir)
        self.rs_dir = os.path.join(self.root_dir, 'result')
        if not os.path.isdir(self.rs_dir):
            os.mkdir(self.rs_dir)
        self.mdl_dir = os.path.join(self.root_dir, 'model')
        if not os.path.isdir(self.mdl_dir):
            os.mkdir(self.mdl_dir)
        self.pic_dir = os.path.join(self.root_dir, 'pics')
        if not os.path.isdir(self.pic_dir):
            os.mkdir(self.pic_dir)

class GetFolder(GetOriFolder):
    def __init__(self):
        GetOriFolder.__init__(self)
        self.data_dir = os.path.join(self.root_dir, 'data')
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        self.log_dir = os.path.join(self.root_dir, 'log')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        self.MBCNN_mdl_dir = os.path.join(self.root_dir, 'MBCNN_mdl')
        if not os.path.isdir(self.MBCNN_mdl_dir):
            os.mkdir(self.MBCNN_mdl_dir)

class GetMdlFolder(GetFolder):
    def __init__(self, host_specie="escherichia coli", target_specie="staphylococcus aureus"):
        GetFolder.__init__(self)
        self.host_abb = abbreviation(host_specie)
        self.tgt_abb = abbreviation(target_specie)
        self.mrg_abb = "%s_%s" % (self.host_abb, self.tgt_abb)
        mdl_dir = self.mdl_dir
        self.host_mdl_dir = os.path.join(mdl_dir, "%s_pMIC" % self.host_abb)
        createFolder(self.host_mdl_dir)
        self.tgt_mdl_dir = os.path.join(mdl_dir, "%s_pMIC" % self.tgt_abb)
        createFolder(self.tgt_mdl_dir)
        self.mrg_mdl_dir = os.path.join(mdl_dir, "%s_pMICR" % self.mrg_abb)
        createFolder(self.mrg_mdl_dir)

class geneCNNMdlFolder(GetMdlFolder):
    def __init__(self, host_specie="escherichia coli", target_specie="staphylococcus aureus"):
        GetMdlFolder.__init__(self, host_specie, target_specie)
        self.cnn_mdl_dir = os.path.join(self.mdl_dir, "CNN", self.mrg_abb)
        createFolder(self.cnn_mdl_dir)
        # supervised model folder:
        self.supervised_mdl_dir = os.path.join(self.cnn_mdl_dir, "supervised")
        createFolder(self.supervised_mdl_dir)
        self.sup_host_mdl_dir = os.path.join(self.supervised_mdl_dir, self.host_abb)
        createFolder(self.sup_host_mdl_dir)
        self.sup_tgt_mdl_dir = os.path.join(self.supervised_mdl_dir, self.tgt_abb)
        createFolder(self.sup_tgt_mdl_dir)
        self.sup_mrg_mdl_dir = os.path.join(self.supervised_mdl_dir, self.mrg_abb)
        createFolder(self.sup_mrg_mdl_dir)
        # unsupervised model folder:
        self.unsupervised_mdl_dir = os.path.join(self.cnn_mdl_dir, "unsupervised")
        createFolder(self.unsupervised_mdl_dir)
        self.unsup_host_mdl_dir = os.path.join(self.unsupervised_mdl_dir, self.host_abb)
        createFolder(self.unsup_host_mdl_dir)
        self.unsup_tgt_mdl_dir = os.path.join(self.unsupervised_mdl_dir, self.tgt_abb)
        createFolder(self.unsup_tgt_mdl_dir)
        self.unsup_mrg_mdl_dir = os.path.join(self.unsupervised_mdl_dir, self.mrg_abb)
        createFolder(self.unsup_mrg_mdl_dir)

def getFastasMicFormMergepath(path):
    info = pd.read_csv(path)
    fastas = info[["ID", "SEQUENCE", "pMICR", "EC_pMIC", "SA_pMIC"]].to_numpy().tolist()
    mic = info[["pMICR", "EC_pMIC", "SA_pMIC"]].to_numpy().tolist()
    return info, fastas, mic

def getOriFastasMicFormMergepath(path):
    info = pd.read_csv(path)
    mics = info[["HOST_MIC", "TARGET_MIC"]].to_numpy().tolist()
    mic_ratio = - np.log10(info["TARGET_MIC"]/info["HOST_MIC"])
    sa_pmic = (- np.log10(info["TARGET_MIC"])).to_numpy().tolist()
    ec_pmic = (- np.log10(info["HOST_MIC"])).to_numpy().tolist()
    mic_ratio = mic_ratio.to_numpy().tolist()
    info["pMICR"] = mic_ratio
    info["EC_pMIC"] = ec_pmic
    info["SA_pMIC"] = sa_pmic
    # info["EC_MIC"] = info["HOST_MIC"]
    # info["SA_MIC"] = info["TARGET_MIC"]
    fastas = info[["ID", "SEQUENCE", "pMICR", "EC_pMIC", "SA_pMIC"]].to_numpy().tolist()
    return info, fastas, mics, mic_ratio

class getMergeFastaColnames():
    def __init__(self, host_specie="escherichia coli", target_specie="staphylococcus aureus"):
        hst_abb, tgt_abb = abbreviation(host_specie), abbreviation(target_specie)
        self.hst_abb, self.tgt_abb = hst_abb, tgt_abb
        self.pMIC_name1, self.pMIC_name2 = "%s_pMIC" % hst_abb, "%s_pMIC" % tgt_abb
        self.MIC_name1, self.MIC_name2 = "%s_MIC" % hst_abb, "%s_MIC" % tgt_abb
        self.pMICR_name = "%s_pMICR_%s" % (tgt_abb, hst_abb)
        self.colnames = ["ID", "SEQUENCE", self.MIC_name1, self.MIC_name2, self.pMIC_name1, self.pMIC_name2, self.pMICR_name]

class getSingleFastaColnames():
    def __init__(self, specie="escherichia coli"):
        abb = abbreviation(specie)
        self.abb = abb
        self.pMIC_name = "%s_pMIC" % abb
        self.MIC_name = "%s_MIC" % abb
        self.colnames = ["ID", "SEQUENCE", self.MIC_name, self.pMIC_name]


class GetOriMergeInfo():
    def __init__(self, host_specie="escherichia coli", target_specie="staphylococcus aureus"):
        self.mrg_path, self.hst_abb, self.tgt_abb, self.mrg_abb = getMergePath(host_specie, target_specie)
        colnames = getMergeFastaColnames(host_specie, target_specie)
        self.colnames = colnames.colnames
        self.mrg_fastas = self.mrg_info
        printDataInfo("Merge %s Info: " % self.mrg_abb, self.mrg_path, self.mrg_info)

def getAllTestPath(host_specie="escherichia coli", target_specie="staphylococcus aureus"):
    mrg_path, hst_abb, tgt_abb, mrg_abb = getMergePath(host_specie, target_specie)
    spl_path = mrg_path.split("/")
    path = os.path.join("/".join(spl_path[:-1]), "all_test-%s" % (spl_path[-1]))
    return path

def getValidationAndTrainPath(file_path):
    spl_path = file_path.split("/")
    tra_path = os.path.join("/".join(spl_path[:-1]), "tra-%s" % (spl_path[-1]))
    val_path = os.path.join("/".join(spl_path[:-1]), "val-%s" % (spl_path[-1]))
    return tra_path, val_path

def getHstAndTgtTrainLocation(host_specie="escherichia coli", target_specie="staphylococcus aureus"):
    mrg_path, hst_abb, tgt_abb, mrg_abb = getMergePath(host_specie, target_specie)
    fld = GetFolder()
    # host train path:
    hst_path = os.path.join(fld.data_dir, "%s.csv" % hst_abb)
    hst_name, train_name = os.path.basename(hst_path), os.path.basename(mrg_path)
    hst_del_mrg_name = "%s-del-all_test-%s" % (hst_name.split(".csv")[0], train_name)
    hst_train_path = os.path.join(fld.data_dir, hst_del_mrg_name)
    # host tra-train, val-train paths:
    tra_hst_train_path, val_hst_train_path = getValidationAndTrainPath(hst_train_path)
    hst_dict = {"path": {"train_path": hst_train_path, "tra_train_path": tra_hst_train_path,
                         "val_train_path": val_hst_train_path}}
    if os.path.isfile(tra_hst_train_path):
        hst_train_fastas = pd.read_csv(hst_train_path)
        printDataInfo("Host %s train Info: " % mrg_abb, hst_train_path, hst_train_fastas)
        tra_hst_train_fastas = pd.read_csv(tra_hst_train_path)
        printDataInfo("Host %s tra-train Info: " % mrg_abb, tra_hst_train_path, tra_hst_train_fastas)
        val_hst_train_fastas = pd.read_csv(val_hst_train_path)
        printDataInfo("Host %s val-train Info: " % mrg_abb, val_hst_train_path, val_hst_train_fastas)
        hst_dict["fastas"] = {"train_fastas": hst_train_fastas, "tra_train_fastas": tra_hst_train_fastas,
                              "val_train_fastas": val_hst_train_fastas}
    # target train path
    tgt_path = os.path.join(fld.data_dir, "%s.csv" % tgt_abb)
    tgt_name, train_name = os.path.basename(tgt_path), os.path.basename(mrg_path)
    tgt_del_mrg_name = "%s-del-all_test-%s" % (tgt_name.split(".csv")[0], train_name)
    tgt_train_path = os.path.join(fld.data_dir, tgt_del_mrg_name)
    # target tra-train, val-train paths:
    tra_tgt_train_path, val_tgt_train_path = getValidationAndTrainPath(tgt_train_path)
    tgt_dict = {"path": {"train_path": tgt_train_path, "tra_train_path": tra_tgt_train_path,
                         "val_train_path": val_tgt_train_path}}
    if os.path.isfile(tra_tgt_train_path):
        tgt_train_fastas = pd.read_csv(tgt_train_path)
        printDataInfo("Target %s train Info: " % mrg_abb, tgt_train_path, tgt_train_fastas)
        tra_tgt_train_fastas = pd.read_csv(tra_tgt_train_path)
        printDataInfo("Target %s tra-train Info: " % mrg_abb, tra_tgt_train_path, tra_tgt_train_fastas)
        val_tgt_train_fastas = pd.read_csv(val_tgt_train_path)
        printDataInfo("Target %s val-train Info: " % mrg_abb, val_tgt_train_path, val_tgt_train_fastas)
        tgt_dict["fastas"] = {"train_fastas": tgt_train_fastas, "tra_train_fastas": tra_tgt_train_fastas,
                              "val_train_fastas": val_tgt_train_fastas}
    return hst_dict, tgt_dict


class GetMergeInfo(GetOriMergeInfo):
    def __init__(self, host_specie="escherichia coli", target_specie="staphylococcus aureus", onlyMrg=False,
                 cdhit=False, x=20, redo_single_del_mrg=False):
        print("####################################### Whole Merge Infos ###########################################")
        GetOriMergeInfo.__init__(self, host_specie, target_specie)
        mrg_dir, merge_name = os.path.dirname(self.mrg_path), os.path.basename(self.mrg_path)
        mrg_dir = os.path.join(mrg_dir)
        path = self.mrg_path
        spl_path = path.split("/")

        if not onlyMrg:
            hst_path, hst_abb = getSpeciePath(host_specie)
            tgt_path, tgt_abb = getSpeciePath(target_specie)
            self.hst_path, self.tgt_path = hst_path, tgt_path
            self.hst_abb, self.tgt_abb = hst_abb, tgt_abb
            self.hst_fastas = pd.read_csv(hst_path)
            # self.hst_fastas = self.hst_info
            self.tgt_fastas = pd.read_csv(tgt_path)
            # self.tgt_fastas = self.tgt_info
            print("####################################### host and target Infos ###########################################")
            print("**************** whole set Infos: ****************")
            printDataInfo("whole %s set info: " % (hst_abb), self.hst_path, self.hst_fastas)
            printDataInfo("whole %s set info: " % (tgt_abb), self.tgt_path, self.tgt_fastas)
            self.hst_del_mrg_path = os.path.join(mrg_dir, "%s-del-%s.csv" % (hst_abb, self.mrg_abb))
            self.tgt_del_mrg_path = os.path.join(mrg_dir, "%s-del-%s.csv" % (tgt_abb, self.mrg_abb))
            if redo_single_del_mrg or not os.path.isfile(self.hst_del_mrg_path):
                self.hst_del_mrg_fastas = handleInfos(self.hst_fastas, self.mrg_fastas).infos1DelInfos2()
                self.hst_del_mrg_fastas.to_csv(self.hst_del_mrg_path, index=False)
            if redo_single_del_mrg or not os.path.isfile(self.tgt_del_mrg_path):
                self.tgt_del_mrg_fastas = handleInfos(self.tgt_fastas, self.mrg_fastas).infos1DelInfos2()
                self.tgt_del_mrg_fastas.to_csv(self.tgt_del_mrg_path, index=False)
            self.hst_del_mrg_fastas = pd.read_csv(self.hst_del_mrg_path)
            self.tgt_del_mrg_fastas = pd.read_csv(self.tgt_del_mrg_path)
            print("**************** single delete intersection Infos: ****************")
            printDataInfo("%s delete %s Info: " % (hst_abb, self.mrg_abb), self.hst_del_mrg_path, self.hst_del_mrg_fastas)
            printDataInfo("%s delete %s Info: " % (tgt_abb, self.mrg_abb), self.tgt_del_mrg_path, self.tgt_del_mrg_fastas)
        # elif not cdhit:
        #     self.train_path = os.path.join(mrg_dir, "train_%s" % merge_name)
        #     spl_path_mrg = self.mrg_path.split("/")
        #     self.train_path = os.path.join("/".join(spl_path_mrg[:-1]), "train-%dPercent-%s" % (100 - x, spl_path_mrg[-1]))
        #     self.test_path = os.path.join("/".join(spl_path_mrg[:-1]), "test-%dPercent-%s" % (x, spl_path_mrg[-1]))
        #     self.tra_train_path, self.val_train_path = getValidationAndTrainPath(self.train_path)
        if cdhit:
            path = os.path.join("/".join(spl_path[:-1]), "all_test-%s" % (spl_path[-1]))
            self.all_test_path = path
            self.train_path = os.path.join(mrg_dir, "train_%s" % merge_name)
            self.tra_train_path, self.val_train_path = getValidationAndTrainPath(self.train_path)
            self.test_path = os.path.join(mrg_dir, "test_%s" % merge_name)
            self.novel_path = os.path.join(mrg_dir, "novel_%s" % merge_name)
            self.train_info = pd.read_csv(self.train_path)
            self.train_fastas = self.train_info
            self.test_info = pd.read_csv(self.test_path)
            self.test_fastas = self.test_info
            self.novel_info = pd.read_csv(self.novel_path)
            self.novel_fastas = self.novel_info
            print("####################################### Merge Infos ###########################################")
            printDataInfo("Merge %s train Info: " % self.mrg_abb, self.train_path, self.train_info)

            if os.path.isfile(self.tra_train_path):
                self.tra_train_fastas = pd.read_csv(self.tra_train_path)
                printDataInfo("Merge %s tra-train Info: " % self.mrg_abb, self.tra_train_path, self.tra_train_fastas)
                self.val_train_fastas = pd.read_csv(self.val_train_path)
                printDataInfo("Merge %s val-train Info: " % self.mrg_abb, self.val_train_path, self.val_train_fastas)
            if os.path.isfile(self.all_test_path):
                self.all_test_fastas = pd.read_csv(self.all_test_path)
                printDataInfo("Merge %s all-test Info: " % self.mrg_abb, self.all_test_path, self.all_test_fastas)
            printDataInfo("Merge %s test Info: " % self.mrg_abb, self.test_path, self.test_info)
            printDataInfo("Merge %s novel Info: " % self.mrg_abb, self.novel_path, self.novel_info)

            if not onlyMrg:
                # get host train and target train
                print("##################################### Host and target Infos #########################################")
                hst_dict, tgt_dict = getHstAndTgtTrainLocation(host_specie, target_specie)
                p = hst_dict["path"]
                self.hst_train_path, self.tra_hst_train_path, self.val_hst_train_path = p["train_path"], p[
                    "tra_train_path"], p["val_train_path"]
                p = tgt_dict["path"]
                self.tgt_train_path, self.tra_tgt_train_path, self.val_tgt_train_path = p["train_path"], p[
                    "tra_train_path"], p["val_train_path"]
                if "fastas" in hst_dict.keys():
                    f = hst_dict["fastas"]
                    self.hst_train_fastas, self.tra_hst_train_fastas, self.val_hst_train_fastas = f["train_fastas"], f[
                        "tra_train_fastas"], f["val_train_fastas"]
                    f = tgt_dict["fastas"]
                    self.tgt_train_fastas, self.tra_tgt_train_fastas, self.val_tgt_train_fastas = f["train_fastas"], f[
                        "tra_train_fastas"], f["val_train_fastas"]
        else:
            train_path = os.path.join("/".join(spl_path[:-1]), "train-%dPercent-%s" % (100 - x, spl_path[-1]))
            test_path = os.path.join("/".join(spl_path[:-1]), "test-%dPercent-%s" % (x, spl_path[-1]))
            self.test_path, self.train_path = test_path, train_path
            self.train_fastas = pd.read_csv(self.train_path)
            self.test_fastas = pd.read_csv(self.test_path)
            print("**************** random select test from Merge: ****************")
            printDataInfo("Merge %s train Info delete corresponding test from whole: " % (self.mrg_abb), self.train_path, self.train_fastas)
            printDataInfo("Merge %s test Info ranomly select %d percent sequences from whole: " % (self.mrg_abb, x), self.test_path, self.test_fastas)
            if not onlyMrg:
                self.tra_train_path, self.val_train_path = getValidationAndTrainPath(train_path)
                if not os.path.isfile(self.tra_train_path):
                    tra_path, val_path = splitValidationFromTrain(self.train_path)
                self.tra_train_fastas = pd.read_csv(self.tra_train_path)
                self.val_train_fastas = pd.read_csv(self.val_train_path)
                printDataInfo("Merge %s train Info delete corresponding test from whole: " % (self.mrg_abb),
                              self.tra_train_path, self.tra_train_fastas)
                printDataInfo("Merge %s train Info delete corresponding test from whole: " % (self.mrg_abb),
                              self.val_train_path, self.val_train_fastas)
                # hst_path, hst_abb = getSpeciePath(host_specie)
                hst_name, test_name = os.path.basename(hst_path), os.path.basename(test_path)
                hst_del_mrg_name = "%s-del-%s" % (hst_name.split(".csv")[0], test_name)
                self.hst_train_path = os.path.join(mrg_dir, hst_del_mrg_name)
                # tgt_path, tgt_abb = getSpeciePath(target_specie)
                tgt_name = os.path.basename(tgt_path)
                tgt_del_mrg_name = "%s-del-%s" % (tgt_name.split(".csv")[0], test_name)
                self.tgt_train_path = os.path.join(mrg_dir, tgt_del_mrg_name)
                self.hst_train_info = pd.read_csv(self.hst_train_path)
                self.hst_train_fastas = self.hst_train_info
                self.tgt_train_info = pd.read_csv(self.tgt_train_path)
                self.tgt_train_fastas = self.tgt_train_info
                printDataInfo("Train of %s is whole %s set delete test set of %s: " % (hst_abb, hst_abb, self.mrg_abb),
                              self.hst_train_path, self.hst_train_fastas)
                printDataInfo("Train of %s is whole %s set delete test set of %s: " % (tgt_abb, tgt_abb, self.mrg_abb),
                              self.tgt_train_path, self.tgt_train_fastas)


class GetDelInfo():
    def __init__(self, host_specie="escherichia coli", target_specie="staphylococcus aureus"):
        self.hst_abb, self.tgt_abb = abbreviation(host_specie), abbreviation(target_specie)
        self.mrg_abb = "%s_mrg_%s" % (self.tgt_abb, self.hst_abb)
        self.hst_del_mrg_path, self.tgt_del_mrg_path = getDelPath(host_specie, target_specie)
        self.hst_del_mrg_info = pd.read_csv(self.hst_del_mrg_path)
        self.tgt_del_mrg_info = pd.read_csv(self.tgt_del_mrg_path)
        self.hst_del_mrg_fastas, self.tgt_del_mrg_fastas = self.hst_del_mrg_info, self.tgt_del_mrg_info
        printDataInfo("%s delete %s Info: " % (self.hst_abb, self.mrg_abb), self.hst_del_mrg_path, self.hst_del_mrg_info)
        printDataInfo("%s delete %s Info: " %(self.tgt_abb, self.mrg_abb), self.tgt_del_mrg_path, self.tgt_del_mrg_info)

def getSpeciePath(specie_name="escherichia coli", threshold=largest_MIC):
    fld = GetFolder()
    abb = abbreviation(specie_name)
    file_name = "%s.csv" % abb
    file_path = os.path.join(fld.data_dir, "MIC%d" % threshold, file_name)
    return file_path, abb


def getMergePath(host_specie="escherichia coli", target_specie="staphylococcus aureus", threshold=largest_MIC):
    fld = GetFolder()
    hst_abb, tgt_abb = abbreviation(host_specie), abbreviation(target_specie)
    mrg_abb = "%s_mrg_%s" % (tgt_abb, hst_abb)
    mrg_path = os.path.join(fld.data_dir, "MIC%d" % threshold, "%s.csv" % mrg_abb)
    return mrg_path, hst_abb, tgt_abb, mrg_abb


def getDelPath(host_specie="escherichia coli", target_specie="staphylococcus aureus"):
    fld = GetFolder()
    hst_abb, tgt_abb = abbreviation(host_specie), abbreviation(target_specie)
    mrg_abb = "%s_mrg_%s" % (tgt_abb, hst_abb)
    hst_del_mrg_name = "%s-del-%s.csv" % (hst_abb, mrg_abb)
    hst_del_mrg_path = os.path.join(fld.data_dir, hst_del_mrg_name)
    tgt_del_mrg_name = "%s-del-%s.csv" % (tgt_abb, mrg_abb)
    tgt_del_mrg_path = os.path.join(fld.data_dir, tgt_del_mrg_name)
    return hst_del_mrg_path, tgt_del_mrg_path
