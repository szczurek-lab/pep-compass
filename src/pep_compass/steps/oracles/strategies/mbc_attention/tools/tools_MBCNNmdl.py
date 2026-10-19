from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    train_test_split,
)  # or StratifiedShuffleSplit
from tools.MultiBranchCNN import *

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    import pandas as pd
    from matplotlib.axes import Axes


def_lrParas = {
    "lr": 1e-4,
    "decay_rate": 0.9,
    "decay_steps": 100,
    "patience": 20,
    "monitor": "loss",
}
cal_lrParas = {
    "lr": 1e-4,
    "decay_rate": 0.86,
    "decay_steps": 50,
    "patience": 20,
    "monitor": "loss",
}
lrParasDict = {"sodium": def_lrParas, "potassium": def_lrParas, "calcium": cal_lrParas}
hyperParaDict = {
    "layer_num": 1,
    "dropOutRate": 0.0,
    "filter_num": 64,
    "whole_tune_epoch": 200,
    "epoch": 1000,
}
lrParas = {
    "lr": 5e-4,
    "decay_rate": 0.92,
    "decay_steps": 25,
    "patience": 15,
    "monitor": "loss",
}

best30_pmic4_uni = [
    "type8raac9glmd3lambda-correlation",
    "QSOrder_lmd4",
    "type5raac15glmd4lambda-correlation",
    "type7raac10glmd3lambda-correlation",
    "type3Braac9glmd3lambda-correlation",
    "type2raac15glmd4lambda-correlation",
    "type14raac10glmd3lambda-correlation",
    "type16raac15glmd4lambda-correlation",
]


def splitDatasets(
    data_dict: Mapping[str, object], test_size: float, random_state: int | None = None
) -> tuple[object, object]:
    """Split datasets.

    :param data_dict: Data dict.
    :type data_dict: Mapping[str, object]
    :param test_size: Test size.
    :type test_size: float
    :param random_state: Random state.
    :type random_state: int | None
    :return: Result of this operation.
    :rtype: tuple[object, object]
    """
    sss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    ft_list = list(data_dict.keys())
    data_ft = data_dict[ft_list[0]]
    X, y = data_ft["img"], data_ft["cls"]
    sss.get_n_splits(X, y)
    train_index, test_index = next(sss.split(X, y))
    test_dict, train_dict = {}, {}
    for ft in ft_list:
        data_ft = data_dict[ft]
        X, y = data_ft["img"], data_ft["cls"]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        test_dict[ft] = {"img": X_test, "cls": y_test}
        train_dict[ft] = {"img": X_train, "cls": y_train}
    return train_dict, test_dict


def add2Datasets(
    data_dict1: Mapping[str, object], data_dict2: Mapping[str, object]
) -> dict[object, object]:
    """Add2datasets.

    :param data_dict1: Data dict1.
    :type data_dict1: Mapping[str, object]
    :param data_dict2: Data dict2.
    :type data_dict2: Mapping[str, object]
    :return: Result of this operation.
    :rtype: dict[object, object]
    """
    data_dict = {}
    for ft in data_dict1:
        data_ft1, data_ft2 = data_dict1[ft], data_dict2[ft]
        # y is (num, ) an array so use np.hstack or np.concatenate, vstack is for matrix
        x1, y1 = data_ft1["img"], data_ft1["cls"]
        x2, y2 = data_ft2["img"], data_ft2["cls"]
        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
        data_dict[ft] = {"img": x, "cls": y}
    return data_dict


def scatterSubEC(
    pred: np.ndarray,
    gt: np.ndarray,
    ax: Axes,
    edge: tuple[int, int] = (-4, 4),
    legend: bool = True,
) -> None:
    """Scatter sub ec.

    :param pred: Pred.
    :type pred: np.ndarray
    :param gt: Gt.
    :type gt: np.ndarray
    :param ax: Ax.
    :type ax: Axes
    :param edge: Edge.
    :type edge: tuple[int, int]
    :param legend: Legend.
    :type legend: bool
    :rtype: None
    """
    x = np.linspace(edge[0], edge[1], (edge[1] - edge[0]) * 100)
    ax.plot(x, x, color="r", linewidth=1, label="Diagonal")
    ax.set_xlim([edge[0], edge[1]])
    ax.set_ylim([edge[0], edge[1]])
    ax.scatter(gt, pred, c="g", alpha=0.3, s=8, label="pMIC value (%d)" % len(gt))
    if legend:
        ax.legend()
    return


def startScatter(
    rep: int, testset: str = "test", target: str = "EC_pMIC", font: int = 10
) -> tuple[object, object]:
    """Start scatter.

    :param rep: Rep.
    :type rep: int
    :param testset: Testset.
    :type testset: str
    :param target: Target.
    :type target: str
    :param font: Font.
    :type font: int
    :return: Result of this operation.
    :rtype: tuple[object, object]
    """
    if testset == "test":
        row, col = 1, rep
    else:
        row, col = rep, rep
    fig, axs = plt.subplots(row, col, sharex=True, sharey=True, squeeze=False)
    space = 1 / 2 / rep
    fig.text(
        0.06,
        0.5,
        "Predict %s" % target,
        fontsize=font + 2,
        fontweight="bold",
        rotation=90,
    )
    fig.text(0.5, 0.06, "True %s" % target, fontsize=font + 2, fontweight="bold")
    for i in range(col):
        fig.text(
            1.9 * space + 1.58 * i * space,
            0.9,
            str(i + 1),
            fontsize=font,
            fontweight="bold",
        )
    if row > 1:
        for i in range(row):
            fig.text(
                0.08,
                1.6 * space + 1.6 * i * space,
                str(i + 1),
                fontsize=font,
                fontweight="bold",
            )
    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    h, w = row * fig.get_figheight(), col * fig.get_figwidth()
    fig.set_figheight(h)
    fig.set_figwidth(w)
    plt.suptitle(testset, fontsize=font + 6, fontweight="bold")
    return fig, axs


def startScatterEC(
    rep: int, testset: str = "test", target: str = "EC_pMIC", font: int = 10
) -> tuple[object, object]:
    """Start scatter ec.

    :param rep: Rep.
    :type rep: int
    :param testset: Testset.
    :type testset: str
    :param target: Target.
    :type target: str
    :param font: Font.
    :type font: int
    :return: Result of this operation.
    :rtype: tuple[object, object]
    """
    row, col = 1, rep
    fig, axs = plt.subplots(row, col, sharex=True, sharey=True)
    space = 1 / 2 / rep
    if col == 1:
        fig.text(
            0.04,
            0.4,
            "Predicted %s" % target,
            fontsize=font + 2,
            fontweight="bold",
            rotation=90,
        )
        fig.text(0.45, 0.02, "True %s" % target, fontsize=font + 2, fontweight="bold")
    else:
        fig.text(
            0.09,
            0.4,
            "Predicted %s" % target,
            fontsize=font + 2,
            fontweight="bold",
            rotation=90,
        )
        fig.text(0.48, 0.02, "True %s" % target, fontsize=font + 2, fontweight="bold")
    if col > 1:
        for i in range(col):
            fig.text(
                1.4 * space + 1.65 * i * space,
                0.9,
                str(i + 1),
                fontsize=font,
                fontweight="bold",
            )
    if row > 1:
        for i in range(row):
            fig.text(
                0.08,
                1.6 * space + 1.6 * i * space,
                str(i + 1),
                fontsize=font,
                fontweight="bold",
            )
    plt.rcParams["font.family"] = "DeJavu Serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    h, w = row * fig.get_figheight(), col * fig.get_figwidth()
    fig.set_figheight(h)
    fig.set_figwidth(w)
    plt.suptitle(testset, fontsize=font + 6, fontweight="bold")
    return fig, axs


def calMeanStdPandas(
    mt_list: Sequence[pd.DataFrame], testset: str = "validation", std_mode: str = "sem"
) -> tuple[pd.DataFrame, object]:
    """:param mt_list:
    :param testset:
    :param std_mode: sem: standard error of the mean, std: standard error
    :return:

    :type mt_list: Sequence[pd.DataFrame]
    :type testset: str
    :type std_mode: str
    :rtype: tuple[pd.DataFrame, object]
    """
    mts = pd.concat(mt_list)
    mean = mts.mean().to_frame().transpose().copy()
    cmd = "mts.%s().to_frame().transpose().copy()" % std_mode
    std = eval(cmd)
    new_mts = pd.concat((mts, mean, std))
    l = len(new_mts)
    test_names = ["%s%d" % (testset, i + 1) for i in range(l - 2)]
    test_names.append("mean %s" % testset)
    test_names.append("%s %s" % (std_mode, testset))
    new_mts.insert(0, "testset", test_names)
    return new_mts, mean


def scatterSub(
    pred: np.ndarray, gt: np.ndarray, ax: Axes, edge: int = 1, legend: bool = True
) -> None:
    """Scatter sub.

    :param pred: Pred.
    :type pred: np.ndarray
    :param gt: Gt.
    :type gt: np.ndarray
    :param ax: Ax.
    :type ax: Axes
    :param edge: Edge.
    :type edge: int
    :param legend: Legend.
    :type legend: bool
    :rtype: None
    """
    x = np.linspace(0, edge, edge * 100)
    ax.plot(x, x, color="r", linewidth=1, label="Diagonal")
    ax.set_xlim([0, edge])
    ax.set_ylim([0, edge])
    ax.scatter(gt, pred, c="g", alpha=0.3, s=8, label="MIC value")
    ax.legend()
    return


def descale(
    output: np.ndarray, bias: float = def_bias, scale: float = def_scale
) -> np.ndarray:
    """Descale.

    :param output: Output.
    :type output: np.ndarray
    :param bias: Bias.
    :type bias: float
    :param scale: Scale.
    :type scale: float
    :return: Result of this operation.
    :rtype: np.ndarray
    """
    new_out = output / scale - bias
    return new_out


def MBCNN3timesValidation(
    ft_name: str = best30_pmic4_uni[:3],
    host_specie: str = "escherichia coli",
    target_specie: str = "staphylococcus aureus",
    data_method: str = "geneMergeFts",
    do_test: bool = True,
    plot_scatter: bool = False,
    font: int = 10,
    onlyMrg: bool = False,
    cdhit: bool = False,
    test_ratio: int = 20,
    val_ratio: int = 20,
    process: str = "host",
    repeat: int = 3,
    test_rep: bool = True,
    target: str = "EC_pMIC",
    lrParas: Mapping[str, object] = def_lrParas,
    hyperParas: Mapping[str, object] = hyperParaDict,
    enable_earlyStop: bool = True,
    threshold: float = largest_MIC,
    bias: float = def_bias,
    scale: float = def_scale,
    mdl_name: str | None = None,
    hist_name: str | None = None,
    log_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mbcnn3times validation.

    :param ft_name: Ft name.
    :type ft_name: str
    :param host_specie: Host specie.
    :type host_specie: str
    :param target_specie: Target specie.
    :type target_specie: str
    :param data_method: Data method.
    :type data_method: str
    :param do_test: Do test.
    :type do_test: bool
    :param plot_scatter: Plot scatter.
    :type plot_scatter: bool
    :param font: Font.
    :type font: int
    :param onlyMrg: Onlymrg.
    :type onlyMrg: bool
    :param cdhit: Cdhit.
    :type cdhit: bool
    :param test_ratio: Test ratio.
    :type test_ratio: int
    :param val_ratio: Val ratio.
    :type val_ratio: int
    :param process: Process.
    :type process: str
    :param repeat: Repeat.
    :type repeat: int
    :param test_rep: Test rep.
    :type test_rep: bool
    :param target: Target.
    :type target: str
    :param lrParas: Lrparas.
    :type lrParas: Mapping[str, object]
    :param hyperParas: Hyperparas.
    :type hyperParas: Mapping[str, object]
    :param enable_earlyStop: Enable earlystop.
    :type enable_earlyStop: bool
    :param threshold: Threshold.
    :type threshold: float
    :param bias: Bias.
    :type bias: float
    :param scale: Scale.
    :type scale: float
    :param mdl_name: Mdl name.
    :type mdl_name: str | None
    :param hist_name: Hist name.
    :type hist_name: str | None
    :param log_name: Log name.
    :type log_name: str | None
    :return: Result of this operation.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    c = -1
    # if MIC  threshold=10 then test_ratio 20, val_ratio 10,
    # if thresthold is 10000 then test_ratio 20, val_ratio 20,
    # names = getMergeFastaColnames(host_specie=host_specie, target_specie=target_specie)
    # target = names.MIC_name1
    fld = GetFolder()
    log_dir = fld.log_dir
    mdl_dir = fld.MBCNN_mdl_dir
    lr_names = "-".join(["%s_%s" % (i, str(lrParas[i])) for i in lrParas])
    hyperPara_names = "-".join(["%s_%s" % (i, str(hyperParas[i])) for i in hyperParas])
    # mdl_dir = os.path
    # log_path = os.path.join(log_dir, log_name)
    use_fts = "best-%dfts" % len(ft_name)
    if not log_name:
        log_name = "temp_multi_branch_cnn.rs"
    log_path = os.path.join(log_dir, "%s-%s" % (use_fts, log_name))
    pkl_path = log_path.split(".")
    pkl_path[-1] = "pkl"
    pkl_path = ".".join(pkl_path)
    if not mdl_name:
        mdl_name = "temp_multi_branch_cnn.mdl"
    mdl_dir = os.path.join(mdl_dir, "tuneLrParas", "GridSearchLossMSLE")
    createFolder(mdl_dir)
    mdl_path = os.path.join(mdl_dir, mdl_name)
    if not hist_name:
        hist_name = "temp_multi_branch_cnn.hist"
    hist_path = os.path.join(mdl_dir, hist_name)
    if data_method == "geneMergeFts":
        data = geneMergeFtsMultiBranch(
            host_specie=host_specie,
            target_specie=target_specie,
            ft_name=ft_name,
            target=target,
            onlyMrg=onlyMrg,
            cdhit=cdhit,
            x=test_ratio,
            process=process,
            test_rep=test_rep,
        )
    if data_method == "geneSingleFts":
        data = geneSingleFtsMultiBranch(
            specie=host_specie, ft_name=ft_name, target=target
        )
    if test_rep:
        mrg_dict = data.mrg_dict
        all_del_mrg_dict = data.all_del_mrg_dict
        rep_t = repeat
        # fts = np.vstack((fts, ft))
    else:
        train_dict = data.train_dict
        test_dict = data.test_dict
        rep_t = 1
    # randomly select test 20% form merge, if
    mean_train_list = []
    mean_val_list = []
    test_list = []
    all_mts_list = []
    if plot_scatter:
        fig_tra, axs_tra = startScatter(
            repeat, testset="tra-train", target=target, font=font
        )
        fig_val, axs_val = startScatter(
            repeat, testset="val-test", target=target, font=font
        )
        fig_test, axs_test = startScatter(
            repeat, testset="test", target=target, font=font
        )
    # do training and testing
    all_rs = {}
    for j in range(rep_t):
        new_mdl_path = "%s.test%d" % (mdl_path, j + 1)
        new_hist_path = "%s.test%d" % (hist_path, j + 1)
        rs_dict = {}
        test_dict = {}
        if test_rep:
            mrg_part_dict, test_dict = splitDatasets(
                mrg_dict, test_size=test_ratio / 100, random_state=None
            )
            # y is (num, ) an array so use np.hstack or np.concatenate, vstack is for matrix
            train_dict = add2Datasets(mrg_part_dict, all_del_mrg_dict)
        mt_list = []
        # test_list = []
        tra_mt_list = []
        for i in range(repeat):
            rs_temp = {}
            tra_dict, val_dict = splitDatasets(
                train_dict, test_size=val_ratio / 100, random_state=None
            )
            CNNMdl = DevelopfitAndSaveCNNMdl(
                tra_dict,
                val_dict,
                "%s.val%d" % (new_mdl_path, i + 1),
                "%s.val%d" % (new_hist_path, j + 1),
                hyperParas,
                lrParas,
                enable_earlyStop,
            )
            tra_x, tra_y = CNNstandardInputOutput(tra_dict)
            val_x, val_y = CNNstandardInputOutput(val_dict)
            # print("val_y: \n", val_y)
            val_pre = CNNMdl.predict(val_x)
            # print("val_pre: \n", val_pre)
            tra_pre = CNNMdl.predict(tra_x)
            tra_y, tra_pre = (
                descale(tra_y, bias=bias, scale=scale),
                descale(tra_pre, bias=bias, scale=scale),
            )
            val_y, val_pre = (
                descale(val_y, bias=bias, scale=scale),
                descale(val_pre, bias=bias, scale=scale),
            )
            tra_mt = calMetrics(tra_y, tra_pre)
            val_mt = calMetrics(val_y, val_pre)
            rs_temp["tra_train"] = {"true": tra_y, "pred": tra_pre, "metrics": tra_mt}
            rs_temp["val_train"] = {"true": val_y, "pred": val_pre, "metrics": val_mt}
            rs_dict[i] = rs_temp
            print("tra_mt: \n", tra_mt)
            print("val_mt: \n", val_mt)
            tra_mt_list.append(tra_mt)
            mt_list.append(val_mt)
            if plot_scatter:
                scatterSub(tra_pre, tra_y, axs_tra[i, j])
                scatterSub(val_pre, val_y, axs_val[i, j])
        tra_mts, tra_mean = calMeanStdPandas(
            tra_mt_list, testset="tra_train", std_mode="std"
        )
        val_mts, val_mean = calMeanStdPandas(
            mt_list, testset="val_train", std_mode="std"
        )
        # all_rs[j] = rs_dict
        mean_train_list.append(tra_mean)
        mean_val_list.append(val_mean)
        mts = pd.concat((tra_mts, val_mts))
        rs_dict["metrics"] = mts
        all_mts_list.append(mts)
        if do_test:
            CNNMdl = DevelopfitAndSaveCNNMdl(
                train_dict,
                test_dict,
                new_mdl_path,
                new_hist_path,
                hyperParas,
                lrParas,
                enable_earlyStop,
            )
            test_x, test_y = CNNstandardInputOutput(test_dict)
            test_pre = CNNMdl.predict(test_x)
            test_y, test_pre = (
                descale(test_y, bias=bias, scale=scale),
                descale(test_pre, bias=bias, scale=scale),
            )
            test_mt = calMetrics(test_y, test_pre)
            test_dict = {"true": test_y, "pred": test_pre, "metrics": test_mt}
            print("test_mt: \n", test_mt)
            test_list.append(test_mt)
            if plot_scatter:
                scatterSub(test_pre, test_y, axs_test[0, j])
        all_rs[j] = {"train": rs_dict, "test": test_dict}
    fig_tra.savefig(
        "scatter_%s_tra-train_80train_EC_MIC%d.png" % (use_fts, threshold),
        dpi=100,
        format="png",
    )
    fig_val.savefig(
        "scatter_%s_val-train_20train_EC_MIC%d.png" % (use_fts, threshold),
        dpi=100,
        format="png",
    )
    fig_test.savefig(
        "scatter_%s_test_20Mrg_EC_MIC%d.png" % (use_fts, threshold),
        dpi=100,
        format="png",
    )
    plt.close("all")
    tra_ave_mts, mean = calMeanStdPandas(
        mean_train_list, testset="tra_train_mean", std_mode="sem"
    )
    val_ave_mts, mean = calMeanStdPandas(
        mean_val_list, testset="val_train_mean", std_mode="sem"
    )
    test_mts, mean = calMeanStdPandas(test_list, testset="test", std_mode="std")
    all_mts = pd.concat(all_mts_list)
    all_mts = pd.concat((all_mts, test_mts))
    all_mts.insert(0, "ft_name", [use_fts] * len(all_mts))
    all_mts.insert(0, "hyper_paras", [hyperPara_names] * len(all_mts))
    all_mts.insert(0, "lr_paras", [lr_names] * len(all_mts))
    ave_mts = pd.concat((tra_ave_mts, val_ave_mts, test_mts))
    ave_mts.insert(0, "ft_name", [use_fts] * len(ave_mts))
    ave_mts.insert(0, "hyper_paras", [hyperPara_names] * len(ave_mts))
    ave_mts.insert(0, "lr_paras", [lr_names] * len(ave_mts))
    all_rs["metrics"] = ave_mts
    all_rs["all metrics"] = all_mts
    all_mts.to_csv("%s.all" % log_path, index=False)
    ave_mts.to_csv("%s.ave" % log_path, index=False)
    write_pkl(all_rs, pkl_path)
    print("all metrics saved to: ", log_path)
    print("all result saved to: ", pkl_path)
    print("ave_mts: \n", ave_mts)
    return ave_mts, all_mts


def MBCNN3timesValidationEC(
    ft_name: str = best30_pmic4_uni[:3],
    specie: str = "escherichia coli",
    do_test: bool = True,
    plot_scatter: bool = False,
    font: int = 9,
    edge: tuple[int, int] = (-4, 4),
    repeat: int = 3,
    target: str = "EC_pMIC",
    lrParas: Mapping[str, object] = def_lrParas,
    hyperParas: Mapping[str, object] = hyperParaDict,
    enable_earlyStop: bool = True,
    threshold: float = largest_MIC,
    bias: float = def_bias,
    scale: float = def_scale,
    mdl_name: str | None = None,
    hist_name: str | None = None,
    log_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mbcnn3times validation ec.

    :param ft_name: Ft name.
    :type ft_name: str
    :param specie: Specie.
    :type specie: str
    :param do_test: Do test.
    :type do_test: bool
    :param plot_scatter: Plot scatter.
    :type plot_scatter: bool
    :param font: Font.
    :type font: int
    :param edge: Edge.
    :type edge: tuple[int, int]
    :param repeat: Repeat.
    :type repeat: int
    :param target: Target.
    :type target: str
    :param lrParas: Lrparas.
    :type lrParas: Mapping[str, object]
    :param hyperParas: Hyperparas.
    :type hyperParas: Mapping[str, object]
    :param enable_earlyStop: Enable earlystop.
    :type enable_earlyStop: bool
    :param threshold: Threshold.
    :type threshold: float
    :param bias: Bias.
    :type bias: float
    :param scale: Scale.
    :type scale: float
    :param mdl_name: Mdl name.
    :type mdl_name: str | None
    :param hist_name: Hist name.
    :type hist_name: str | None
    :param log_name: Log name.
    :type log_name: str | None
    :return: Result of this operation.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    c = -1
    # if MIC  threshold=10 then test_ratio 20, val_ratio 10,
    # if thresthold is 10000 then test_ratio 20, val_ratio 20,
    # names = getMergeFastaColnames(host_specie=host_specie, target_specie=target_specie)
    # target = names.MIC_name1
    fld = GetFolder()
    log_dir = fld.log_dir
    mdl_dir = fld.MBCNN_mdl_dir
    lr_names = "-".join(["%s_%s" % (i, str(lrParas[i])) for i in lrParas])
    hyperPara_names = "-".join(["%s_%s" % (i, str(hyperParas[i])) for i in hyperParas])
    # mdl_dir = os.path
    # log_path = os.path.join(log_dir, log_name)
    use_fts = "best-%dfts" % len(ft_name)
    if not log_name:
        log_name = "temp_multi_branch_cnn.rs"
    log_path = os.path.join(log_dir, "%s-%s" % (use_fts, log_name))
    pkl_path = log_path.split(".")
    pkl_path[-1] = "pkl"
    pkl_path = ".".join(pkl_path)
    if not mdl_name:
        mdl_name = "temp_multi_branch_cnn.mdl"
    mdl_dir = os.path.join(mdl_dir, "tuneLrParas", "GridSearchLossMSLE")
    createFolder(mdl_dir)
    mdl_path = os.path.join(mdl_dir, mdl_name)
    if not hist_name:
        hist_name = "temp_multi_branch_cnn.hist"
    hist_path = os.path.join(mdl_dir, hist_name)
    rep_t = 1
    # randomly select test 20% form merge, if
    mean_train_list = []
    mean_val_list = []
    test_list, val_list, tra_list = [], [], []
    all_mts_list = []
    if plot_scatter:
        fig_tra, axs_tra = startScatterEC(
            repeat, testset="tra-train", target=target, font=font
        )
        fig_val, axs_val = startScatterEC(
            repeat, testset="val-train", target=target, font=font
        )
        fig_test, axs_test = startScatterEC(
            repeat, testset="test", target=target, font=font
        )
    # do training and testing
    all_rs = {}
    for j in range(repeat):
        new_mdl_path = "%s.test%d" % (mdl_path, j + 1)
        new_hist_path = "%s.test%d" % (hist_path, j + 1)
        rs_dict = {}
        data = geneSingleFtsMultiBranch(ft_name, target, specie=specie, rep=j)
        test_dict, train_dict, val_dict, tra_dict = (
            data.test_dict,
            data.train_dict,
            data.val_dict,
            data.tra_dict,
        )
        mt_list = []
        tra_mt_list = []
        for i in range(rep_t):
            rs_temp = {}
            CNNMdl = DevelopfitAndSaveCNNMdl(
                tra_dict,
                val_dict,
                "%s.val%d" % (new_mdl_path, i + 1),
                "%s.val%d" % (new_hist_path, j + 1),
                hyperParas,
                lrParas,
                enable_earlyStop,
            )
            tra_x, tra_y = CNNstandardInputOutput(tra_dict)
            val_x, val_y = CNNstandardInputOutput(val_dict)
            # print("val_y: \n", val_y)
            val_pre = CNNMdl.predict(val_x)
            # print("val_pre: \n", val_pre)
            tra_pre = CNNMdl.predict(tra_x)
            tra_y, tra_pre = (
                descale(tra_y, bias=bias, scale=scale),
                descale(tra_pre, bias=bias, scale=scale),
            )
            val_y, val_pre = (
                descale(val_y, bias=bias, scale=scale),
                descale(val_pre, bias=bias, scale=scale),
            )
            tra_mt = calMetrics(tra_y, tra_pre)
            val_mt = calMetrics(val_y, val_pre)
            rs_temp["tra_train"] = {"true": tra_y, "pred": tra_pre, "metrics": tra_mt}
            rs_temp["val_train"] = {"true": val_y, "pred": val_pre, "metrics": val_mt}
            rs_dict[i] = rs_temp
            print("tra_mt: \n", tra_mt)
            print("val_mt: \n", val_mt)
            tra_mt_list.append(tra_mt)
            mt_list.append(val_mt)
            if plot_scatter and rep_t == 1:
                scatterSubEC(tra_pre, tra_y, axs_tra[j], edge=edge)
                scatterSubEC(val_pre, val_y, axs_val[j], edge=edge)
        if rep_t > 1:
            tra_mts, tra_mean = calMeanStdPandas(
                tra_mt_list, testset="tra_train", std_mode="std"
            )
            val_mts, val_mean = calMeanStdPandas(
                mt_list, testset="val_train", std_mode="std"
            )
            mean_train_list.append(tra_mean)
            mean_val_list.append(val_mean)
            mts = pd.concat((tra_mts, val_mts))
            rs_dict["metrics"] = mts
            all_mts_list.append(mts)
        if rep_t == 1:
            tra_list.append(tra_mt)
            val_list.append(val_mt)
        if do_test:
            CNNMdl = DevelopfitAndSaveCNNMdl(
                train_dict,
                test_dict,
                new_mdl_path,
                new_hist_path,
                hyperParas,
                lrParas,
                enable_earlyStop,
            )
            test_x, test_y = CNNstandardInputOutput(test_dict)
            test_pre = CNNMdl.predict(test_x)
            test_y, test_pre = (
                descale(test_y, bias=bias, scale=scale),
                descale(test_pre, bias=bias, scale=scale),
            )
            test_mt = calMetrics(test_y, test_pre)
            test_dict = {"true": test_y, "pred": test_pre, "metrics": test_mt}
            print("test_mt: \n", test_mt)
            test_list.append(test_mt)
            if plot_scatter:
                scatterSubEC(test_pre, test_y, axs_test[j], edge=edge)
        all_rs[j] = {"train": rs_dict, "test": test_dict}
    createFolder("MBC_scatter")
    fig_tra.savefig(
        "./MBC_scatter/scatter_%s_tra-train_90train_EC_pMIC4_MBC.png" % (use_fts),
        dpi=100,
        format="png",
    )
    fig_val.savefig(
        "./MBC_scatter/scatter_%s_val-train_10train_EC_pMIC4_MBC.png" % (use_fts),
        dpi=100,
        format="png",
    )
    fig_test.savefig(
        "./MBC_scatter/scatter_%s_test_10All_EC_pMIC4_MBC.png" % (use_fts),
        dpi=100,
        format="png",
    )
    print(
        "all figs saved to: \n\t",
        "./MBC_scatter/scatter_%s_test_10All_EC_pMIC4_MBC.png" % (use_fts),
    )
    plt.close("all")
    test_mts, mean = calMeanStdPandas(test_list, testset="test", std_mode="std")
    if rep_t > 1:
        tra_ave_mts, mean = calMeanStdPandas(
            mean_train_list, testset="tra_train_mean", std_mode="sem"
        )
        val_ave_mts, mean = calMeanStdPandas(
            mean_val_list, testset="val_train_mean", std_mode="sem"
        )
    if rep_t == 1:
        tra_mts, tra_mean = calMeanStdPandas(
            tra_list, testset="tra_train", std_mode="std"
        )
        val_mts, val_mean = calMeanStdPandas(
            val_list, testset="val_train", std_mode="std"
        )
        tra_ave_mts, val_ave_mts = tra_mts, val_mts
        mean_train_list.append(tra_mean)
        mean_val_list.append(val_mean)
        mts = pd.concat((tra_mts, val_mts))
        rs_dict["metrics"] = mts
        all_mts_list.append(mts)
    all_mts = pd.concat(all_mts_list)
    all_mts = pd.concat((all_mts, test_mts))
    all_mts.insert(0, "ft_name", [use_fts] * len(all_mts))
    all_mts.insert(0, "hyper_paras", [hyperPara_names] * len(all_mts))
    all_mts.insert(0, "lr_paras", [lr_names] * len(all_mts))
    ave_mts = pd.concat((tra_ave_mts, val_ave_mts, test_mts))
    ave_mts.insert(0, "ft_name", [use_fts] * len(ave_mts))
    ave_mts.insert(0, "hyper_paras", [hyperPara_names] * len(ave_mts))
    ave_mts.insert(0, "lr_paras", [lr_names] * len(ave_mts))
    all_rs["metrics"] = ave_mts
    all_rs["all metrics"] = all_mts
    all_mts.to_csv("%s.all" % log_path, index=False)
    ave_mts.to_csv("%s.ave" % log_path, index=False)
    write_pkl(all_rs, pkl_path)
    print("all metrics saved to: ", log_path)
    print("all result saved to: ", pkl_path)
    print("ave_mts: \n", ave_mts)
    return ave_mts, all_mts


def DevelopFinalMdl(
    ft_name: str,
    specie: str = "escherichia coli",
    repeat: int = 3,
    target: str = "EC_pMIC",
    lrParas: Mapping[str, object] = def_lrParas,
    do_test: bool = True,
    hyperParas: Mapping[str, object] = hyperParaDict,
    enable_earlyStop: bool = True,
    threshold: float = largest_MIC,
    bias: float = def_bias,
    scale: float = def_scale,
    mdl_name: str | None = None,
    hist_name: str | None = None,
    log_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Develop final mdl.

    :param ft_name: Ft name.
    :type ft_name: str
    :param specie: Specie.
    :type specie: str
    :param repeat: Repeat.
    :type repeat: int
    :param target: Target.
    :type target: str
    :param lrParas: Lrparas.
    :type lrParas: Mapping[str, object]
    :param do_test: Do test.
    :type do_test: bool
    :param hyperParas: Hyperparas.
    :type hyperParas: Mapping[str, object]
    :param enable_earlyStop: Enable earlystop.
    :type enable_earlyStop: bool
    :param threshold: Threshold.
    :type threshold: float
    :param bias: Bias.
    :type bias: float
    :param scale: Scale.
    :type scale: float
    :param mdl_name: Mdl name.
    :type mdl_name: str | None
    :param hist_name: Hist name.
    :type hist_name: str | None
    :param log_name: Log name.
    :type log_name: str | None
    :return: Result of this operation.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    c = -1
    # if MIC  threshold=10 then test_ratio 20, val_ratio 10,
    # if thresthold is 10000 then test_ratio 20, val_ratio 20,
    # names = getMergeFastaColnames(host_specie=host_specie, target_specie=target_specie)
    # target = names.MIC_name1
    fld = GetFolder()
    log_dir = fld.log_dir
    mdl_dir = fld.MBCNN_mdl_dir
    lr_names = "-".join(["%s_%s" % (i, str(lrParas[i])) for i in lrParas])
    hyperPara_names = "-".join(["%s_%s" % (i, str(hyperParas[i])) for i in hyperParas])
    # mdl_dir = os.path
    # log_path = os.path.join(log_dir, log_name)
    use_fts = "best-%dfts" % len(ft_name)
    if not log_name:
        log_name = "temp_multi_branch_cnn.rs"
    log_path = os.path.join(log_dir, "%s-%s" % (use_fts, log_name))
    pkl_path = log_path.split(".")
    pkl_path[-1] = "pkl"
    pkl_path = ".".join(pkl_path)
    if not mdl_name:
        mdl_name = "temp_multi_branch_cnn.mdl"
    mdl_dir = os.path.join(mdl_dir, "tuneLrParas", "GridSearchLossMSLE")
    createFolder(mdl_dir)
    mdl_path = os.path.join(mdl_dir, mdl_name)
    if not hist_name:
        hist_name = "temp_multi_branch_cnn.hist"
    hist_path = os.path.join(mdl_dir, hist_name)
    rep_t = 1
    # randomly select test 20% form merge, if
    mean_train_list = []
    mean_val_list = []
    test_list, val_list, tra_list = [], [], []
    all_mts_list = []
    # do training and testing
    all_rs = {}
    for j in range(repeat):
        new_mdl_path = "%s.test%d" % (mdl_path, j + 1)
        new_hist_path = "%s.test%d" % (hist_path, j + 1)
        rs_dict = {}
        data = geneSingleFtsMultiBranch(ft_name, target, specie=specie, rep=j)
        test_dict, train_dict, val_dict, tra_dict = (
            data.test_dict,
            data.train_dict,
            data.val_dict,
            data.tra_dict,
        )
        mt_list = []
        tra_mt_list = []
        for i in range(rep_t):
            rs_temp = {}
            CNNMdl = DevelopfitAndSaveCNNMdl(
                tra_dict,
                val_dict,
                "%s.val%d" % (new_mdl_path, i + 1),
                "%s.val%d" % (new_hist_path, j + 1),
                hyperParas,
                lrParas,
                enable_earlyStop,
            )
            tra_x, tra_y = CNNstandardInputOutput(tra_dict)
            val_x, val_y = CNNstandardInputOutput(val_dict)
            # print("val_y: \n", val_y)
            val_pre = CNNMdl.predict(val_x)
            # print("val_pre: \n", val_pre)
            tra_pre = CNNMdl.predict(tra_x)
            tra_y, tra_pre = (
                descale(tra_y, bias=bias, scale=scale),
                descale(tra_pre, bias=bias, scale=scale),
            )
            val_y, val_pre = (
                descale(val_y, bias=bias, scale=scale),
                descale(val_pre, bias=bias, scale=scale),
            )
            tra_mt = calMetrics(tra_y, tra_pre)
            val_mt = calMetrics(val_y, val_pre)
            rs_temp["tra_train"] = {"true": tra_y, "pred": tra_pre, "metrics": tra_mt}
            rs_temp["val_train"] = {"true": val_y, "pred": val_pre, "metrics": val_mt}
            rs_dict[i] = rs_temp
            print("tra_mt: \n", tra_mt)
            print("val_mt: \n", val_mt)
            tra_mt_list.append(tra_mt)
            mt_list.append(val_mt)
        if rep_t > 1:
            tra_mts, tra_mean = calMeanStdPandas(
                tra_mt_list, testset="tra_train", std_mode="std"
            )
            val_mts, val_mean = calMeanStdPandas(
                mt_list, testset="val_train", std_mode="std"
            )
            mean_train_list.append(tra_mean)
            mean_val_list.append(val_mean)
            mts = pd.concat((tra_mts, val_mts))
            rs_dict["metrics"] = mts
            all_mts_list.append(mts)
        if rep_t == 1:
            tra_list.append(tra_mt)
            val_list.append(val_mt)
        if do_test:
            CNNMdl = DevelopfitAndSaveCNNMdl(
                train_dict,
                test_dict,
                new_mdl_path,
                new_hist_path,
                hyperParas,
                lrParas,
                enable_earlyStop,
            )
            test_x, test_y = CNNstandardInputOutput(test_dict)
            test_pre = CNNMdl.predict(test_x)
            test_y, test_pre = (
                descale(test_y, bias=bias, scale=scale),
                descale(test_pre, bias=bias, scale=scale),
            )
            test_mt = calMetrics(test_y, test_pre)
            test_dict = {"true": test_y, "pred": test_pre, "metrics": test_mt}
            print("test_mt: \n", test_mt)
            test_list.append(test_mt)
        all_rs[j] = {"train": rs_dict, "test": test_dict}
    createFolder("MBC_scatter")
    print(
        "all figs saved to: \n\t",
        "./MBC_scatter/scatter_%s_test_10All_EC_pMIC4_MBC.png" % (use_fts),
    )
    plt.close("all")
    test_mts, mean = calMeanStdPandas(test_list, testset="test", std_mode="std")
    if rep_t > 1:
        tra_ave_mts, mean = calMeanStdPandas(
            mean_train_list, testset="tra_train_mean", std_mode="sem"
        )
        val_ave_mts, mean = calMeanStdPandas(
            mean_val_list, testset="val_train_mean", std_mode="sem"
        )
    if rep_t == 1:
        tra_mts, tra_mean = calMeanStdPandas(
            tra_list, testset="tra_train", std_mode="std"
        )
        val_mts, val_mean = calMeanStdPandas(
            val_list, testset="val_train", std_mode="std"
        )
        tra_ave_mts, val_ave_mts = tra_mts, val_mts
        mean_train_list.append(tra_mean)
        mean_val_list.append(val_mean)
        mts = pd.concat((tra_mts, val_mts))
        rs_dict["metrics"] = mts
        all_mts_list.append(mts)
    all_mts = pd.concat(all_mts_list)
    all_mts = pd.concat((all_mts, test_mts))
    all_mts.insert(0, "ft_name", [use_fts] * len(all_mts))
    all_mts.insert(0, "hyper_paras", [hyperPara_names] * len(all_mts))
    all_mts.insert(0, "lr_paras", [lr_names] * len(all_mts))
    ave_mts = pd.concat((tra_ave_mts, val_ave_mts, test_mts))
    ave_mts.insert(0, "ft_name", [use_fts] * len(ave_mts))
    ave_mts.insert(0, "hyper_paras", [hyperPara_names] * len(ave_mts))
    ave_mts.insert(0, "lr_paras", [lr_names] * len(ave_mts))
    all_rs["metrics"] = ave_mts
    all_rs["all metrics"] = all_mts
    all_mts.to_csv("%s.all" % log_path, index=False)
    ave_mts.to_csv("%s.ave" % log_path, index=False)
    write_pkl(all_rs, pkl_path)
    print("all metrics saved to: ", log_path)
    print("all result saved to: ", pkl_path)
    print("ave_mts: \n", ave_mts)
    return ave_mts, all_mts


if __name__ == "__main__":
    best30_pmic4 = [
        "type8raac16glmd3lambda-correlation",
        "type2raac15glmd3lambda-correlation",
        "type11raac12glmd3lambda-correlation",
        "type8raac12glmd3lambda-correlation",
        "type5raac15glmd3lambda-correlation",
        "QSOrder_lmd3",
        "type13raac17glmd1lambda-correlation",
        "type5raac15glmd0g-gap",
        "type10raac13glmd5lambda-correlation",
        "type3Braac12glmd1lambda-correlation",
        "type16raac9glmd3lambda-correlation",
        "QSOrder_lmd4",
        "type8raac12glmd2lambda-correlation",
        "type1raac13glmd0g-gap",
        "type10raac13glmd2lambda-correlation",
        "type8raac11glmd1lambda-correlation",
        "type8raac15glmd3lambda-correlation",
        "type10raac11glmd3lambda-correlation",
        "type11raac10glmd2lambda-correlation",
        "type11raac12glmd0g-gap",
        "type8raac15glmd0g-gap",
        "type7raac10glmd1lambda-correlation",
        "type8raac11glmd2lambda-correlation",
        "type8raac11glmd3lambda-correlation",
        "PAAC_lmd0",
        "type10raac11glmd2lambda-correlation",
        "type7raac18glmd2lambda-correlation",
        "type16raac15glmd3lambda-correlation",
        "type3Braac12glmd2lambda-correlation",
        "type7raac18glmd1lambda-correlation",
    ]
    log_name = "EC_MIC10000_pMIC_scale6_bias3_8_rep3-layer1-dropout0.4-filter32.rs"
    # server = 3
    # s, e = 5 * (server-1) + 1, 5 * server + 1
    # print("MBCNN best_%d to best_%d average metrics: \n" % (s, e), rs)
    fld = GetFolder()
    log_dir = fld.log_dir
    name = "best14_MBCNN-%s" % (log_name)
    path = os.path.join(log_dir, name)

    ave_mts, all_mts = MBCNN3timesValidation(
        ft_name=best30_pmic4[:14],
        host_specie="escherichia coli",
        target_specie="staphylococcus aureus",
        data_method="geneMergeFts",
        do_test=False,
        plot_scatter=True,
        font=10,
        onlyMrg=False,
        cdhit=False,
        test_ratio=20,
        val_ratio=10,
        process="host",
        repeat=1,
        test_rep=True,
        target="EC_pMIC",
        lrParas=def_lrParas,
        hyperParas=hyperParaDict,
        mdl_name=None,
        hist_name=None,
        log_name=log_name,
        enable_earlyStop=True,
    )
