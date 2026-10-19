from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf
from keras import Sequential, layers, losses, metrics, models
from keras import backend as K
from keras.models import Model
from tensorflow import keras
from tools.base import *
from tools.features import *
from tools.preProcessed import *

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import numpy as np
    import pandas as pd


def addZeroCols(df: pd.DataFrame, ktuple: int = 2) -> pd.DataFrame:
    """Add zero cols.

    :param df: Df.
    :type df: pd.DataFrame
    :param ktuple: Ktuple.
    :type ktuple: int
    :return: Result of this operation.
    :rtype: pd.DataFrame
    """
    row, col = df.shape
    df = df.copy()
    p = ktuple
    add_col = int(np.power(np.ceil(np.power(col, 1 / p)), p) - col)
    for i in range(add_col):
        col_name = "zero%d" % i
        df[col_name] = [0] * row
    return df


def MLftset2ftImgAndLabel(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame | np.ndarray | list[float] | None]:
    """Mlftset2ft img and label.

    :param df: Df.
    :type df: pd.DataFrame
    :return: Result of this operation.
    :rtype: dict[str, pd.DataFrame | np.ndarray | list[float] | None]
    """
    colname = df.columns.to_list()
    if "default" in colname:
        label = df["default"].to_list()
        new_df = df.drop(["default"], 1)
    else:
        label = None
        new_df = df
    dict = {"img": new_df.copy(), "cls": label}
    return dict


def CNNftset2ftImgAndLabel(
    df: pd.DataFrame, ktuple: int = 2
) -> dict[str, pd.DataFrame | np.ndarray | list[float] | None]:
    """Cnnftset2ft img and label.

    :param df: Df.
    :type df: pd.DataFrame
    :param ktuple: Ktuple.
    :type ktuple: int
    :return: Result of this operation.
    :rtype: dict[str, pd.DataFrame | np.ndarray | list[float] | None]
    """
    if "default" in df.columns.to_list():
        label = df["default"].to_list()
        row = len(df)
        label = np.array(label).reshape((row, 1))
        label = label.astype(dtype=np.float32)
        new_df = df.drop(["default"], axis=1)
    else:
        label = None
        new_df = df.copy()
    # add zero columns to a square or 3 power
    new_df = addZeroCols(new_df, ktuple=ktuple)
    new_df = new_df.to_numpy()
    row, col = new_df.shape
    img_width = int(np.power(col, 1 / ktuple))
    if ktuple == 2:
        new_df = np.reshape(new_df, (row, img_width, img_width, 1))
    if ktuple == 3:
        new_df = np.reshape(new_df, (row, img_width, img_width, img_width))
    new_df = new_df.astype(dtype=np.float32)
    dict = {"img": new_df, "cls": label}
    return dict


def CNNimportFtsDataSetsPoNe(
    fastas: Sequence[Sequence[str]],
    ft_list: Sequence[str],
    target: str,
    bias: float = def_bias,
    scale: float = def_scale,
) -> dict[object, object]:
    """Cnnimport fts data sets po ne.

    :param fastas: Sequence records in the layout expected by the feature encoder.
    :type fastas: Sequence[Sequence[str]]
    :param ft_list: Ft list.
    :type ft_list: Sequence[str]
    :param target: Target.
    :type target: str
    :param bias: Bias.
    :type bias: float
    :param scale: Scale.
    :type scale: float
    :return: Result of this operation.
    :rtype: dict[object, object]
    """
    sets = {}
    if type(ft_list) is not list:
        ft_list = [ft_list]
    for ft_whole_name in ft_list:
        df = geneXYDataFrame(fastas, ft_whole_name, target, bias=bias, scale=scale)
        dict = CNNftset2ftImgAndLabel(df)
        sets[ft_whole_name] = dict
    return sets


class geneSingleFtsMultiBranch:
    def __init__(
        self,
        ft_name: str,
        target: str,
        specie: str = "escherichia coli",
        del_novel: bool = False,
        cdhit_flag: bool = False,
        rep: int = 0,
        test_validation: bool = False,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param ft_name: Ft name.
        :type ft_name: str
        :param target: Target.
        :type target: str
        :param specie: Specie.
        :type specie: str
        :param del_novel: Del novel.
        :type del_novel: bool
        :param cdhit_flag: Cdhit flag.
        :type cdhit_flag: bool
        :param rep: Rep.
        :type rep: int
        :param test_validation: Test validation.
        :type test_validation: bool
        :rtype: None
        """
        spe = GetSpecieInfo(
            specie_name=specie, del_novel=del_novel, cdhit_flag=cdhit_flag, rep=rep
        )
        if test_validation:
            self.tra_dict = CNNimportFtsDataSetsPoNe(
                spe.tra_train_fastas, ft_name, target
            )
            self.val_dict = CNNimportFtsDataSetsPoNe(
                spe.val_train_fastas, ft_name, target
            )
        self.train_dict = CNNimportFtsDataSetsPoNe(spe.train_fastas, ft_name, target)
        self.test_dict = CNNimportFtsDataSetsPoNe(spe.test_fastas, ft_name, target)
        all_dict = {}
        for ft in self.train_dict:
            t_dict = {}
            all_img = np.concatenate(
                (self.train_dict[ft]["img"], self.test_dict[ft]["img"])
            )
            all_cls = np.concatenate(
                (self.train_dict[ft]["cls"], self.test_dict[ft]["cls"])
            )
            t_dict["img"] = all_img
            t_dict["cls"] = all_cls
            all_dict[ft] = t_dict
        self.all_dict = all_dict


class geneMergeFtsMultiBranch:
    def __init__(
        self,
        ft_name: str,
        target: str,
        host_specie: str = "escherichia coli",
        target_specie: str = "staphylococcus aureus",
        onlyMrg: bool = False,
        cdhit: bool = False,
        x: int = 20,
        test_rep: bool = True,
        process: str = "host",
    ) -> None:
        """Initialize this object with the declared configuration.

        :param ft_name: Ft name.
        :type ft_name: str
        :param target: Target.
        :type target: str
        :param host_specie: Host specie.
        :type host_specie: str
        :param target_specie: Target specie.
        :type target_specie: str
        :param onlyMrg: Onlymrg.
        :type onlyMrg: bool
        :param cdhit: Cdhit.
        :type cdhit: bool
        :param x: X.
        :type x: int
        :param test_rep: Test rep.
        :type test_rep: bool
        :param process: Process.
        :type process: str
        :rtype: None
        """
        g = GetMergeInfo(
            host_specie=host_specie,
            target_specie=target_specie,
            onlyMrg=onlyMrg,
            cdhit=cdhit,
            x=x,
        )
        if process == "merge":
            self.tra_dict = CNNimportFtsDataSetsPoNe(
                g.tra_train_fastas, ft_name, target
            )
            self.val_dict = CNNimportFtsDataSetsPoNe(
                g.val_train_fastas, ft_name, target
            )
            self.train_dict = CNNimportFtsDataSetsPoNe(g.train_fastas, ft_name, target)
            self.all_test_dict = CNNimportFtsDataSetsPoNe(
                g.all_test_fastas, ft_name, target
            )
            self.test_dict = CNNimportFtsDataSetsPoNe(g.test_fastas, ft_name, target)
            self.novel_dict = CNNimportFtsDataSetsPoNe(g.novel_fastas, ft_name, target)
        if process == "host" and not onlyMrg:
            if test_rep:
                self.mrg_dict = CNNimportFtsDataSetsPoNe(g.mrg_fastas, ft_name, target)
                self.all_del_mrg_dict = CNNimportFtsDataSetsPoNe(
                    g.hst_del_mrg_fastas, ft_name, target
                )
            else:
                self.train_dict = CNNimportFtsDataSetsPoNe(
                    g.hst_train_fastas, ft_name, target
                )
                self.test_dict = CNNimportFtsDataSetsPoNe(
                    g.test_fastas, ft_name, target
                )
            if cdhit:
                self.tra_dict = CNNimportFtsDataSetsPoNe(
                    g.tra_hst_train_fastas, ft_name, target
                )
                self.val_dict = CNNimportFtsDataSetsPoNe(
                    g.val_hst_train_fastas, ft_name, target
                )
                self.all_test_dict = CNNimportFtsDataSetsPoNe(
                    g.all_test_fastas, ft_name, target
                )
                self.novel_dict = CNNimportFtsDataSetsPoNe(
                    g.novel_fastas, ft_name, target
                )
        if process == "target" and not onlyMrg:
            if test_rep:
                self.mrg_dict = CNNimportFtsDataSetsPoNe(g.mrg_fastas, ft_name, target)
                self.all_del_mrg_dict = CNNimportFtsDataSetsPoNe(
                    g.tgt_del_mrg_fastas, ft_name, target
                )
            else:
                self.train_dict = CNNimportFtsDataSetsPoNe(
                    g.tgt_train_fastas, ft_name, target
                )
                self.test_dict = CNNimportFtsDataSetsPoNe(
                    g.test_fastas, ft_name, target
                )
            if cdhit:
                self.tra_dict = CNNimportFtsDataSetsPoNe(
                    g.tra_tgt_train_fastas, ft_name, target
                )
                self.val_dict = CNNimportFtsDataSetsPoNe(
                    g.val_tgt_train_fastas, ft_name, target
                )
                self.all_test_dict = CNNimportFtsDataSetsPoNe(
                    g.all_test_fastas, ft_name, target
                )
                self.novel_dict = CNNimportFtsDataSetsPoNe(
                    g.novel_fastas, ft_name, target
                )


def import1FtData(
    path: str | Path,
    ft_whole_name: str = "type1raac10",
    subtype: dict[str, int] = {"g-gap": 0, "lambda-correlation": 4},
    ktuple: int = 2,
    gap_lambda: int = 1,
    nlag: int = 4,
    lambdaValue: int = 4,
    target: str | None = None,
) -> pd.DataFrame:
    """Import1ft data.

    :param path: Input or output path used by this operation.
    :type path: str | Path
    :param ft_whole_name: Ft whole name.
    :type ft_whole_name: str
    :param subtype: Subtype.
    :type subtype: dict[str, int]
    :param ktuple: Ktuple.
    :type ktuple: int
    :param gap_lambda: Gap lambda.
    :type gap_lambda: int
    :param nlag: Nlag.
    :type nlag: int
    :param lambdaValue: Lambdavalue.
    :type lambdaValue: int
    :param target: Target.
    :type target: str | None
    :return: Result of this operation.
    :rtype: pd.DataFrame
    """
    if "type" in ft_whole_name:
        raactype = int(ft_whole_name.split("raac")[-1])
        if "Ktuple" in ft_whole_name:
            ft_name = ft_whole_name.split("Ktuple")[0]
        else:
            ft_name = ft_whole_name.split("raac")[0]
        df = genePsekraac(
            path, ft_name, raactype, subtype, ktuple, gap_lambda, class_val=target
        )
    else:
        ft_name = ft_whole_name
        df = GeneIfeature(path, ft_name, nlag, lambdaValue, class_val=target)
    return df


def CNNimportFtsDataSets(
    path: str | Path, ft_list: Sequence[str], target: str | None = None
) -> dict[object, object]:
    """Cnnimport fts data sets.

    :param path: Input or output path used by this operation.
    :type path: str | Path
    :param ft_list: Ft list.
    :type ft_list: Sequence[str]
    :param target: Target.
    :type target: str | None
    :return: Result of this operation.
    :rtype: dict[object, object]
    """
    sets = {}
    if type(ft_list) is not list:
        ft_list = [ft_list]
    for ft_whole_name in ft_list:
        print("processing: %s" % ft_whole_name)
        df = import1FtData(path, ft_whole_name, class_val=target)
        dict = CNNftset2ftImgAndLabel(df)
        sets[ft_whole_name] = dict
    return sets


def MLstandardInputOutput(
    sets: Mapping[str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]],
) -> pd.DataFrame:
    """Mlstandard input output.

    :param sets: Sets.
    :type sets: Mapping[str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]]
    :return: Result of this operation.
    :rtype: pd.DataFrame
    """
    c = -1
    for ft_type in sets:
        c += 1
        if c == 0:
            ft = sets[ft_type]["img"]
        else:
            temp = sets[ft_type]["img"]
            ft = pd.concat([ft, temp], axis=1)
    row, col = ft.shape
    col_names = ["ft%d" % i for i in range(col)]
    ft.columns = col_names
    if sets[ft_type]["cls"] != None:
        ft["default"] = sets[ft_type]["cls"]
    return ft


def CNNstandardInputOutput(sets: object) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Cnnstandard input output.

    :param sets: Sets.
    :type sets: object
    :return: Result of this operation.
    :rtype: tuple[list[np.ndarray], np.ndarray | None]
    """
    X = []
    for ft_type in sets:
        x = sets[ft_type]["img"]
        X.append(x)
    Y = sets[ft_type]["cls"]
    return X, Y


def PCC(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Pcc.

    :param y_true: Y true.
    :type y_true: tf.Tensor
    :param y_pred: Y pred.
    :type y_pred: tf.Tensor
    :return: Result of this operation.
    :rtype: tf.Tensor
    """
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)
    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)
    return tf.keras.backend.mean(fsp * fst) / (devP * devT)


def R2(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """R2.

    :param y_true: Y true.
    :type y_true: tf.Tensor
    :param y_pred: Y pred.
    :type y_pred: tf.Tensor
    :return: Result of this operation.
    :rtype: tf.Tensor
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def cnnBranch(
    in_shape: Sequence[int],
    hyperParas: Mapping[str, object],
    attention_local: bool = True,
) -> tf.Tensor:
    """Cnn branch.

    :param in_shape: In shape.
    :type in_shape: Sequence[int]
    :param hyperParas: Hyperparas.
    :type hyperParas: Mapping[str, object]
    :param attention_local: Attention local.
    :type attention_local: bool
    :return: Result of this operation.
    :rtype: tf.Tensor
    """
    cnn = keras.Sequential(
        [
            layers.BatchNormalization(input_shape=in_shape),
            layers.Conv2D(
                hyperParas["filter_num"], (3, 3), activation="relu", padding="same"
            ),
        ]
    )
    if attention_local:
        key = layers.MaxPooling2D((1, 1), padding="same")(cnn.output)
        query = layers.MaxPooling2D((1, 1), padding="same")(cnn.output)
        value = layers.MaxPooling2D((1, 1), padding="same")(cnn.output)
        attention = layers.Attention()([query, value, key])
        out = attention
    else:
        out = cnn.output
    x = layers.MaxPooling2D((2, 2), padding="same")(out)
    x = layers.Dropout(hyperParas["dropOutRate"])(x)
    return x


def cnnArch(
    sets: Mapping[str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]],
    hyperParas: Mapping[str, object],
    opt: str | keras.optimizers.Optimizer,
    attention: bool = True,
) -> keras.Model:
    """Cnn arch.

    :param sets: Sets.
    :type sets: Mapping[str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]]
    :param hyperParas: Hyperparas.
    :type hyperParas: Mapping[str, object]
    :param opt: Opt.
    :type opt: str | keras.optimizers.Optimizer
    :param attention: Attention.
    :type attention: bool
    :return: Result of this operation.
    :rtype: keras.Model
    """
    mdls, mdls_out, mdls_in = [], [], []
    for key in sets:
        in_shape = sets[key]["img"].shape[1:]
        model = models.Sequential()
        for l in range(hyperParas["layer_num"]):
            model.add(layers.BatchNormalization(input_shape=in_shape))
            model.add(
                layers.Conv2D(
                    hyperParas["filter_num"], (3, 3), activation="relu", padding="same"
                )
            )  # , input_shape=in_shape))
            model.add(layers.MaxPooling2D((2, 2), padding="same"))
            model.add(layers.Dropout(hyperParas["dropOutRate"]))
        model.add(layers.Flatten())
        # model.add(layers.Dense(64, activation='relu'))
        # mdls.append(model)
        mdls_out.append(model.output)
        mdls_in.append(model.input)
    if len(sets.keys()) > 1:
        merge = layers.Add()(mdls_out)
        merge = layers.Dense(32, activation="relu")(merge)
        merge = layers.Dense(1, activation="sigmoid")(merge)
        newMdl = models.Model(mdls_in, merge)
    else:
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        newMdl = model
    newMdl.compile(optimizer=opt, loss="mean_squared_logarithmic_error")
    CNNMdl = newMdl
    return CNNMdl


def cnnBranchTest(in_shape: Sequence[int], attention_local: bool = True) -> keras.Model:
    """Cnn branch test.

    :param in_shape: In shape.
    :type in_shape: Sequence[int]
    :param attention_local: Attention local.
    :type attention_local: bool
    :return: Result of this operation.
    :rtype: keras.Model
    """
    mdl = models.Sequential()
    cnn = keras.Sequential(
        [
            layers.BatchNormalization(input_shape=in_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        ]
    )
    pool = keras.Sequential(
        [
            layers.MaxPooling2D((2, 2), padding="same"),
            layers.Dropout(0.4),
        ]
    )
    mdl.add(cnn)
    out = mdl.output
    shape = mdl.output_shape
    if attention_local:
        key = layers.Conv2D(32, (1, 1), padding="same")(out)
        query = layers.Conv2D(32, (1, 1), padding="same")(out)
        value = layers.Conv2D(32, (1, 1), padding="same")(out)
        key = layers.Reshape((shape[1] * shape[2], shape[3]))(key)
        query = layers.Reshape((shape[1] * shape[2], shape[3]))(query)
        value = layers.Reshape((shape[1] * shape[2], shape[3]))(value)
        scores = tf.matmul(key, query, transpose_b=True)
        distribution = tf.nn.softmax(scores)
        attention = tf.matmul(distribution, value)
        attention = layers.Reshape((shape[1], shape[2], shape[3]))(attention)
        x = layers.Add()([out, attention])
        x = pool(x)
        new_mdl = models.Model(mdl.input, x)
    else:
        new_mdl = mdl.add(pool)
    return new_mdl


def cnnArchtest(
    sets: Mapping[str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]],
    hyperParas: Mapping[str, object],
    opt: str | keras.optimizers.Optimizer,
    attention_global: bool = True,
    attention_local: bool = True,
) -> keras.Model:
    """Cnn archtest.

    :param sets: Sets.
    :type sets: Mapping[str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]]
    :param hyperParas: Hyperparas.
    :type hyperParas: Mapping[str, object]
    :param opt: Opt.
    :type opt: str | keras.optimizers.Optimizer
    :param attention_global: Attention global.
    :type attention_global: bool
    :param attention_local: Attention local.
    :type attention_local: bool
    :return: Result of this operation.
    :rtype: keras.Model
    """
    mdls, mdls_out, mdls_in = [], [], []
    branch = len(sets.keys())
    for key in sets:
        in_shape = sets[key]["img"].shape[1:]
        branches = []
        model = models.Sequential()
        for l in range(1):
            CNN = cnnBranchTest(in_shape, attention_local=attention_local)
            model.add(CNN)
        model.add(layers.Flatten())
        model.add(layers.Dense(64))
        if branch > 1 and attention_global:
            model.add(layers.Reshape((64, 1)))
        mdls_out.append(model.output)
        mdls_in.append(model.input)
    if branch > 1:
        if attention_global:
            merge = layers.Concatenate(axis=2)(mdls_out)
            scores = tf.matmul(merge, merge, transpose_a=True)
            distribution = tf.nn.softmax(scores)
            attention = tf.matmul(merge, distribution)
            merge = layers.Add()([merge, attention])
        else:
            merge = layers.Concatenate(axis=1)(mdls_out)
        merge = layers.Flatten()(merge)
        merge = layers.Dense(32, activation="relu")(merge)
        merge = layers.Dense(1, activation="sigmoid")(merge)
        newMdl = models.Model(mdls_in, merge)
    else:
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        newMdl = model
    newMdl.compile(optimizer=opt, loss="mean_squared_logarithmic_error")
    CNNMdl = newMdl
    return CNNMdl


def DevelopfitAndSaveCNNMdl(
    sets: Mapping[str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]],
    val_sets: Mapping[
        str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]
    ],
    mdl_path: str | Path,
    hist_path: str | Path,
    hyperParas: Mapping[str, object],
    lrParas: Mapping[str, object],
    enable_earlyStop: bool = True,
) -> keras.Model:
    """Developfit and save cnnmdl.

    :param sets: Sets.
    :type sets: Mapping[str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]]
    :param val_sets: Val sets.
    :type val_sets: Mapping[str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]]
    :param mdl_path: Mdl path.
    :type mdl_path: str | Path
    :param hist_path: Hist path.
    :type hist_path: str | Path
    :param hyperParas: Hyperparas.
    :type hyperParas: Mapping[str, object]
    :param lrParas: Lrparas.
    :type lrParas: Mapping[str, object]
    :param enable_earlyStop: Enable earlystop.
    :type enable_earlyStop: bool
    :return: Result of this operation.
    :rtype: keras.Model
    """
    patience, monitor = lrParas["patience"], lrParas["monitor"]
    decay_rate, decay_steps = lrParas["decay_rate"], lrParas["decay_steps"]
    lr = lrParas["lr"]
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate
    )
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    CNNMdl = cnnArchtest(sets, hyperParas, opt)
    X, Y = CNNstandardInputOutput(sets)
    x_val, y_val = CNNstandardInputOutput(val_sets)
    if val_sets == None:
        if enable_earlyStop:
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=patience
            )  # , baseline=baseline)
            history = CNNMdl.fit(
                X,
                Y,
                epochs=hyperParas["epoch"],
                validation_split=0.2,
                callbacks=[callback],
            )
        else:
            history = CNNMdl.fit(X, Y, epochs=hyperParas["epoch"], validation_split=0.2)
    else:
        if enable_earlyStop:
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=patience
            )  # , baseline=baseline)
            history = CNNMdl.fit(
                X,
                Y,
                epochs=hyperParas["epoch"],
                validation_data=(x_val, y_val),
                callbacks=[callback],
            )
        else:
            history = CNNMdl.fit(
                X, Y, epochs=hyperParas["epoch"], validation_data=(x_val, y_val)
            )
    CNNMdl.save(mdl_path)
    print("save CNN mdl to: %s" % mdl_path)
    write_pkl(history.history, hist_path)
    print("save CNN history to: %s" % hist_path)
    return CNNMdl


def predictSequenceFromSaveKerasMdl(
    test_path: str | Path,
    Mdl_path: str | Path,
    testRs_path: str | Path,
    ft_list: Sequence[str],
    target: str,
) -> tuple[object, object]:
    """Predict sequence from save keras mdl.

    :param test_path: Test path.
    :type test_path: str | Path
    :param Mdl_path: Mdl path.
    :type Mdl_path: str | Path
    :param testRs_path: Testrs path.
    :type testRs_path: str | Path
    :param ft_list: Ft list.
    :type ft_list: Sequence[str]
    :param target: Target.
    :type target: str
    :return: Result of this operation.
    :rtype: tuple[object, object]
    """
    sets = CNNimportFtsDataSets(test_path, ft_list, class_val=target)
    names, seqs, lens, records = readFastaYan(test_path)
    X, Y = CNNstandardInputOutput(sets)
    mdl_name = os.path.basename(Mdl_path)
    CNNMdl = keras.models.load_model(Mdl_path)
    pred = CNNMdl.predict(X)
    df = pd.DataFrame(pred[:, 1], index=names, columns=[mdl_name.split(".")[0]])
    df.to_csv(testRs_path)
    print("save %s prediction to: %s" % (test_path, testRs_path))
    return Y, pred


lrParas = {
    "lr": 5e-4,
    "decay_rate": 0.92,
    "decay_steps": 25,
    "patience": 15,
    "monitor": "loss",
}
hyperParaDict = {"layer_num": 1, "dropOutRate": 0.4, "filter_num": 32, "epoch": 200}
best14 = [
    "type8raac9glmd3lambda-correlation",
    "type8raac7glmd3lambda-correlation",
    "QSOrder_lmd4",
    "QSOrder_lmd3",
    "QSOrder_lmd2",
    "QSOrder_lmd1",
    "QSOrder_lmd0",
    "type5raac15glmd4lambda-correlation",
    "type7raac10glmd3lambda-correlation",
    "type5raac8glmd2lambda-correlation",
    "type3Braac9glmd3lambda-correlation",
    "type2raac15glmd4lambda-correlation",
    "type2raac8glmd2lambda-correlation",
    "type8raac14glmd1lambda-correlation",
]


# develop final model
def DevelopfitAndSaveMBCAttenionMdl(
    test_sets: Mapping[
        str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]
    ] = None,
    mdl_path: str | Path | None = None,
    hyperParas: Mapping[str, object] = hyperParaDict,
    lrParas: Mapping[str, object] = lrParas,
    enable_earlyStop: bool = True,
) -> keras.Model:
    # get train fastas
    """Developfit and save mbcattenion mdl.

    :param test_sets: Test sets.
    :type test_sets: Mapping[str, Mapping[str, np.ndarray | pd.DataFrame | list[float] | None]]
    :param mdl_path: Mdl path.
    :type mdl_path: str | Path | None
    :param hyperParas: Hyperparas.
    :type hyperParas: Mapping[str, object]
    :param lrParas: Lrparas.
    :type lrParas: Mapping[str, object]
    :param enable_earlyStop: Enable earlystop.
    :type enable_earlyStop: bool
    :return: Result of this operation.
    :rtype: keras.Model
    """
    fld = GetFolder()
    whole_path = os.path.join(fld.data_dir, "EC.csv")
    train_fastas = pd.read_csv(whole_path)

    # generate features
    train_sets = CNNimportFtsDataSetsPoNe(
        train_fastas, ft_list=best14, target="EC_pMIC"
    )
    # train_X, train_Y = CNNstandardInputOutput(train_sets)

    # train path
    if mdl_path == None:
        mdl_path = os.path.join(fld.mdl_dir, "whole_train.mdl")
        mdl_path = getUnexistedName(mdl_path, file_type=".mdl")

    patience, monitor = lrParas["patience"], lrParas["monitor"]
    decay_rate, decay_steps = lrParas["decay_rate"], lrParas["decay_steps"]
    lr = lrParas["lr"]
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate
    )
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    CNNMdl = cnnArchtest(train_sets, hyperParas, opt)
    X, Y = CNNstandardInputOutput(train_sets)

    if test_sets == None:
        if enable_earlyStop:
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=patience
            )  # , baseline=baseline)
            history = CNNMdl.fit(
                X,
                Y,
                epochs=hyperParas["epoch"],
                validation_split=0.2,
                callbacks=[callback],
            )
        else:
            history = CNNMdl.fit(X, Y, epochs=hyperParas["epoch"], validation_split=0.2)
    else:
        x_test, y_test = CNNstandardInputOutput(test_sets)
        if enable_earlyStop:
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=patience
            )  # , baseline=baseline)
            history = CNNMdl.fit(
                X,
                Y,
                epochs=hyperParas["epoch"],
                validation_data=(x_test, y_test),
                callbacks=[callback],
            )
        else:
            history = CNNMdl.fit(
                X, Y, epochs=hyperParas["epoch"], validation_data=(x_test, y_test)
            )
    CNNMdl.save(mdl_path)
    print("save CNN mdl to: %s" % mdl_path)
    return CNNMdl
