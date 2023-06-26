
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures, PowerTransformer
from sklearn.metrics import make_scorer, f1_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, recall_score, precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator

import matplotlib.pyplot as plt

from scipy.stats import f_oneway, kruskal, chi2_contingency
import scikit_posthocs as sp


from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NeighbourhoodCleaningRule, OneSidedSelection

import json
from itertools import product


def minha_metrica(y_true, y_pred):
    threshold = 0.5  # Defina o threshold desejado

    # Calcule o true positive rate para o threshold dado
    tp = np.sum((y_true == 1) & (y_pred >= threshold))
    fn = np.sum((y_true == 1) & (y_pred < threshold))
    tpr = tp / (tp + fn)

    # Calcule o true negative rate para o threshold dado
    tn = np.sum((y_true == 0) & (y_pred < threshold))
    fp = np.sum((y_true == 0) & (y_pred >= threshold))
    tnr = tn / (tn + fp)

    # Calcule o produto dos passos 1 e 2
    product = tpr * tnr

    # Retorne a raiz quadrada do passo 3
    return np.sqrt(product)


def geo_score(y_true, y_pred):
    # Calcule o true positive rate para o threshold dado
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn)

    # Calcule o true negative rate para o threshold dado
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tnr = tn / (tn + fp)

    # Calcule o produto dos passos 1 e 2
    product = tpr * tnr

    # Retorne a raiz quadrada do passo 3
    return np.sqrt(product)


def get_scores(y_true_tr, y_pred_tr, y_true_ts, y_pred_ts, scorers=[geo_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score]):
    nomes, vtest, vtrain = [], [], []

    fpr, tpr, thresholds = roc_curve(y_true_tr, y_pred_tr)
    distances = (fpr - 0)**2 + (tpr - 1)**2
    index = distances.argmin()
    corte = thresholds[index]
    fprts, tprts, thresholdsts = roc_curve(y_true_ts, y_pred_ts)

    for sc in scorers:
        if sc.__name__ in ["roc_auc_curve"]:
            vtest.append(sc(y_true_ts, y_pred_ts))
            vtrain.append(sc(y_true_tr, y_pred_tr))
            nomes.append(sc.__name__)
        elif sc.__name__ in ["minha_metrica_c"]:
            vtest.append(sc(y_true_ts, y_pred_ts, threshold=corte))
            vtrain.append(sc(y_true_tr, y_pred_tr, threshold=corte))
            nomes.append(sc.__name__)
        else:
            vtest.append(sc(y_true_ts, y_pred_ts >= corte))
            vtrain.append(sc(y_true_tr, y_pred_tr >= corte))
            nomes.append(sc.__name__)

    metricas = pd.DataFrame(
        {"metrica": nomes, "valor no treino": vtrain, "valor no teste": vtest})
    roc_curve_train = {"fpr": fpr, "tpr": tpr,
                       "thresholds": thresholds, "corte": corte}
    roc_curve_test = {"fpr": fprts, "tpr": tprts, "thresholds": thresholdsts}
    cm1 = confusion_matrix(y_pred=y_pred_ts >= corte, y_true=y_true_ts)
    cm2 = confusion_matrix(y_pred=y_pred_ts >= corte,
                           y_true=y_true_ts, normalize='true')
    cm = pd.DataFrame({"pred_0": [cm1[0][0], cm1[1][0]], "pred_1": [cm1[0][1], cm1[1][1]], "predn_0": [
                      cm2[0][0], cm2[1][0]], "predn_1": [cm2[0][1], cm2[1][1]]}, index=["true 0", "true_1"])
    res = {"metricas": metricas, "roc_curve_train": roc_curve_train, "roc_curve_test": roc_curve_test,
           "melhor": [fpr[index], tpr[index], corte], "confusion_matrix": cm}
    return res


def meu_enconder(data, columns, target, split=False, rnd_state=None):
    mapa = {}
    if split:
        data_train, data_test = train_test_split(
            data, test_size=0.2, stratify=data[target], random_state=rnd_state)
        for cl in columns:
            # contagemp = data_train.groupby(cl)[target].value_counts(normalize=True).unstack().fillna(0)
            contagemp = data_train.groupby(cl)[target].value_counts(
                normalize=True).unstack().fillna(0)
            # contagemp["log"] = np.log(contagemp[1]/contagemp[0])
            idx = contagemp.sort_values(by=1, ascending=False).index
            mapeamento = {v: i for i, v in enumerate(idx)}
            mapeamento_inverso = {i: v for i, v in enumerate(idx)}
            data_train[cl] = data_train[cl].map(mapeamento)
            data_test[cl] = data_test[cl].map(mapeamento)
            mapa[cl] = {"mapa": mapeamento, "mapa_inverso": mapeamento_inverso}
            data_test = data_test.dropna()
        return {"train": data_train, "test": data_test, "mapas": mapa}
    else:
        ndata = data.copy()
        for cl in columns:
            contagemp = data.groupby(cl)[target].value_counts(
                normalize=True).unstack().fillna(0)
            # contagemp["log"] = np.log(contagemp[1]/contagemp[0])
            idx = contagemp.sort_values(by=1, ascending=False).index
            mapeamento = {v: i for i, v in enumerate(idx)}
            mapeamento_inverso = {i: v for i, v in enumerate(idx)}
            ndata[cl] = data[cl].map(mapeamento).astype(int)
            mapa[cl] = {"mapa": mapeamento, "mapa_inverso": mapeamento_inverso}
        return {"data": ndata, "mapas": mapa}


def categorizar(data, columns):
    ndata = data.copy()
    intervalos = {}
    for cl in columns:
        if data[cl].dtypes in [np.int64, np.int32, np.float64]:
            categories = pd.qcut(data[cl], 10, duplicates='drop')
            labels, unicos = pd.factorize(categories, sort=True)
            mapa = {f'{i}': [un.left, un.right] for i, un in enumerate(unicos)}
            intervalos[cl] = mapa
            ndata[f"{cl}_cat"] = labels
    return {"data": ndata, "intervalos": intervalos}


def significancia(data, predictors, target, alpha=0.1):
    pval, tval, sigval = [], [], []
    myvars = list(predictors.keys())
    for cl in myvars:
        if predictors[cl]:
            contingency_table = pd.crosstab(data[cl], data[target])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            pval.append(p)
            tval.append(chi2)
            sigval.append(p < alpha)
        else:
            groups = []
            gpstat = data.groupby(target)
            for key in gpstat.groups.keys():
                groups.append(gpstat.get_group(key)[cl])
            f_statistic, p = kruskal(*groups)
            pval.append(p)
            tval.append(f_statistic)
            sigval.append(p < alpha)
    significantvar = list([myvars[i] for i, v in enumerate(sigval) if v])
    stats = pd.DataFrame({"variable": myvars, "test-value": tval,
                         "p-value": pval, "significance": sigval})
    return {"stats": stats, "significantes": significantvar}


def simulador(estimator, data, predictors, target, nsim, metricas=[geo_score, f1_score, accuracy_score, roc_auc_score]):
    metricasval = np.zeros((nsim, len(metricas)))
    truepos, trueneg = [], []
    res = {}
    for i in range(nsim):
        data_train, data_test = train_test_split(
            data, test_size=0.2, stratify=data[target])
        X_train, y_train = data_train[predictors], data_train[target]
        X_test, y_test = data_test[predictors], data_test[target]

        bests = estimator.fit(X_train, y_train)
        try:
            y_pred_ts = bests.predict_proba(X_test)[:, 1]
            y_pred_tr = bests.predict_proba(X_train)[:, 1]
        except:
            y_pred_ts = bests.decision_function(X_test)
            y_pred_tr = bests.decision_function(X_train)

        fpr, tpr, thresholds = roc_curve(y_train, y_pred_tr)
        distances = (fpr - 0)**2 + (tpr - 1)**2
        index = distances.argmin()
        corte = thresholds[index]
        cm = confusion_matrix(y_pred=y_pred_ts >= corte,
                              y_true=y_test, normalize='true')
        truepos.append(cm[1][1])
        trueneg.append(cm[0][0])
        for j, mtr in enumerate(metricas):
            if mtr.__name__ in ["roc_auc_curve"]:
                metricasval[i, j] = mtr(y_test, y_pred_ts)
            else:
                metricasval[i, j] = mtr(y_test, y_pred_ts >= corte)

    res["tpr"] = truepos
    res["tnr"] = trueneg
    for j, mtr in enumerate(metricas):
        res[mtr.__name__] = metricasval[:, j]
    return res


def minha_anova(tabela, alpha=0.05):
    h_statistic, p_value = kruskal(*tabela)
    if p_value > alpha:
        return {"stats": h_statistic, "p_value": p_value}

    else:
        pvals = sp.posthoc_conover(tabela, p_adjust='holm').to_numpy()
        ddif = {}
        for i in range(pvals.shape[0]):
            dff, deq = [], []
            for j in range(pvals.shape[1]):
                if pvals[i, j] < alpha:
                    dff.append(j)
                else:
                    deq.append(j)
            ddif[i] = {"igual_idx": deq, "diferente_idx": dff}

        return {"stats": ddif, "p_value": pvals}


def simulador_cv(estimator, data, predictors, target, categoricalvar, nsim, metricas=[geo_score, f1_score, accuracy_score, roc_auc_score]):
    metricasval = np.zeros((nsim, len(metricas)))
    truepos, trueneg = [], []
    res = {}
    data = data[predictors+[target]]

    for i in range(nsim):
        data_train, data_test = train_test_split(
            data, test_size=0.2, stratify=data[target], random_state=None)
        encodar = [cl for cl in predictors if categoricalvar[cl]]
        data_train, data_test = feat_transform(
            data_train, data_test, encodar, target, categoricalvar)
        data_test = data_test.dropna()

        search = minha_cross_val(estimator, data_train, predictors, target)
        best_idx = search["best_idx"]
        bests = search['estimators'][best_idx]

        y_pred_ts = predicao(bests, data_test[predictors])
        y_pred_tr = predicao(bests, data_train[predictors])

        fpr, tpr, thresholds = roc_curve(data_train[target], y_pred_tr)
        distances = (fpr - 0)**2 + (tpr - 1)**2
        index = distances.argmin()
        corte = thresholds[index]
        cm = confusion_matrix(y_pred=y_pred_ts >= corte,
                              y_true=data_test[target], normalize='true')
        truepos.append(cm[1][1])
        trueneg.append(cm[0][0])
        for j, mtr in enumerate(metricas):
            if mtr.__name__ in ["roc_auc_curve"]:
                metricasval[i, j] = mtr(data_test[target], y_pred_ts)
            else:
                metricasval[i, j] = mtr(data_test[target], y_pred_ts >= corte)

    res["tpr"] = truepos
    res["tnr"] = trueneg
    for j, mtr in enumerate(metricas):
        res[mtr.__name__] = metricasval[:, j]
    return res


def agrupa_predicoes(estimator, data_train, data_test, predictors, target, categoricalvar, corte, alpha=0.05):
    X_train, y_train = data_train[predictors], data_train[target]

    y_pred_train = predicao(estimator, X_train)

    data_train["compara"] = ((y_pred_train >= corte) == y_train).astype(int)
    res_stat = {}
    pred_sep = []
    for cl in predictors:
        if categoricalvar[cl]:
            contingency_table = pd.crosstab(
                data_train[cl], data_train["compara"])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            res_stat[cl] = [p, p < alpha]
            if p < alpha:
                pred_sep.append(cl)
        else:
            groups = []
            gpstat = data_train.groupby("compara")
            for k in gpstat.groups.keys():
                groups.append(gpstat.get_group(k)[cl])
            f_statistic, p = kruskal(*groups)
            res_stat[cl] = [p, p < alpha]
            if p < alpha:
                pred_sep.append(cl)

    pred_sep.append("compara")

    # pred_sep = ["month","no_exa_scheduled_previous_year","compara"]
    normalizador = data_train[pred_sep[:-1]].max().to_numpy()
    # normalizador = np.ones(shape=(len(pred_sep[:-1]),))
    g_tr_comp = data_train[pred_sep].groupby("compara")
    train_0_m = g_tr_comp.get_group(0).mean().to_numpy()[:-1]/normalizador
    train_1_m = g_tr_comp.get_group(1).mean().to_numpy()[:-1]/normalizador

    separa = []
    for y in data_test.index:
        x = data_test[pred_sep[:-1]].loc[y, :].to_numpy()/normalizador
        n0 = np.linalg.norm(x-train_0_m, 4)
        n1 = np.linalg.norm(x-train_1_m, 4)
        separa.append(0*(n1 >= n0)+1*(n1 < n0))

    data_test["separa"] = separa

    X_train_0 = data_train.groupby("compara").get_group(0)[predictors]
    X_train_1 = data_train.groupby("compara").get_group(1)[predictors]

    y_train_0 = data_train.groupby("compara").get_group(0)[target]
    y_train_1 = data_train.groupby("compara").get_group(1)[target]

    X_test_0 = data_test.groupby("separa").get_group(0)[predictors]
    X_test_1 = data_test.groupby("separa").get_group(1)[predictors]

    y_test_0 = data_test.groupby("separa").get_group(0)[target]
    y_test_1 = data_test.groupby("separa").get_group(1)[target]

    y_pred_train_0 = predicao(estimator, X_train_0)
    y_pred_train_1 = predicao(estimator, X_train_1)

    y_pred_test_0 = predicao(estimator, X_test_0)
    y_pred_test_1 = predicao(estimator, X_test_1)

    roc_esc_ts_1 = roc_auc_score(y_test_1, y_pred_test_1)
    roc_esc_ts_0 = roc_auc_score(y_test_0, y_pred_test_0)

    roc_esc_tr_1 = roc_auc_score(y_train_1, y_pred_train_1)
    roc_esc_tr_0 = roc_auc_score(y_train_0, y_pred_train_0)

    fpr_tr0, tpr_tr0, _ = roc_curve(y_train_0, y_pred_train_0)
    fpr_tr1, tpr_tr1, _ = roc_curve(y_train_1, y_pred_train_1)

    fpr_ts1, tpr_ts1, _ = roc_curve(y_test_1, y_pred_test_1)
    fpr_ts0, tpr_ts0, _ = roc_curve(y_test_0, y_pred_test_0)

    curvas_roc = {"train_0": [fpr_tr0, tpr_tr0], "train_1": [
        fpr_tr1, tpr_tr1], "test_0": [fpr_ts0, tpr_ts0], "test_1": [fpr_ts1, tpr_ts1]}
    metrs = {"train_0": roc_esc_tr_0, "train_1": roc_esc_tr_1,
             "test_0": roc_esc_ts_0, "test_1": roc_esc_ts_1}
    return {"data": {"train": data_train, "test": data_test}, "curva_roc": curvas_roc, "metricas": metrs, "separador": pred_sep}


def salvarpipes(nome, mypipe, filename):
    try:
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}
        # bests.named_steps["modelo"].__str__().split("(")[0]
        # bests.named_steps["modelo"].get_params()

    dicio = {}
    for name, step in mypipe.named_steps.items():
        # dicio[name] =  step.__str__().replace("\n","").replace(" ","")
        dicio[name] = {"nome": json.dumps(step.__str__().split(
            "(")[0]), "parametros": json.dumps(step.get_params())}

    existing_data[nome] = dicio
    with open(filename, 'w') as file:
        json.dump(existing_data, file)


def predicao(estimator, X):
    if hasattr(estimator, 'predict_proba'):
        return estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, 'decision_function'):
        return estimator.decision_function(X)
    else:
        raise AttributeError(
            "Estimator does not have predict_proba or decision_function method.")


def feat_transform(data_train, data_test, columns, target, categoricalvar):
    for cl in columns:
        lf = LinearizarFeat()
        if categoricalvar[cl]:
            data_train[cl] = lf.fit_transform(
                data_train[cl], data_train[target])
            data_test[cl] = lf.transform(data_test[cl])
    return data_train, data_test


class LinearizarFeat(BaseEstimator):
    def __init__(self, params={}):
        self.params = params

    def fit(self, X, y):
        dados = pd.DataFrame({"predictor": X, "target": y})
        cont = dados.groupby("predictor")["target"].value_counts(
            normalize=True).unstack().fillna(0)
        idx, probs = list(cont.index), list(cont[1])
        y0, y1 = min(probs), max(probs)
        m = y1-y0
        xb = [np.abs((probs[id]-y0)/m) for id in range(len(probs))]
        mapa = {idx[i]: xb[i] for i in range(len(probs))}
        self.params = mapa
        return self

    def transform(self, X):
        mapa = self.params
        X_transformado = []
        for v in X:
            if v in mapa.keys():
                X_transformado.append(mapa[v])
            else:
                X_transformado.append(np.nan)
        return X_transformado

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def minha_cross_valx(estimador,
                     dados,
                     predictors,
                     target,
                     n_splits=5,
                     metrica=roc_auc_score,
                     rdn_state=None):
    X, y = dados[predictors], dados[target]
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=rdn_state)
    scores_tr, scores_ts, estfit = [], [], []
    for train_index, test_index in skf.split(X, y):
        data_tr, data_ts = dados.iloc[train_index,
                                      :], dados.iloc[test_index, :]
        data_ts = data_ts.dropna()
        X_train_fold, X_test_fold = data_tr[predictors], data_ts[predictors]
        y_train_fold, y_test_fold = data_tr[target], data_ts[target]
        est = estimador.fit(X_train_fold, y_train_fold)
        y_pred_tr = predicao(estimador, X_train_fold)
        y_pred_ts = predicao(estimador, X_test_fold)
        if metrica.__name__ in ["roc_auc_score"]:
            scores_tr.append(metrica(y_train_fold, y_pred_tr))
            scores_ts.append(metrica(y_test_fold, y_pred_ts))
        else:
            fpr, tpr, thresholds = roc_curve(y_train_fold, y_pred_tr)
            distances = (fpr - 0)**2 + (tpr - 1)**2
            idx_mtr = distances.argmin()
            corte = thresholds[idx_mtr]
            scores_tr.append(metrica(y_train_fold, y_pred_tr >= corte))
            scores_ts.append(metrica(y_test_fold, y_pred_ts >= corte))

        estfit.append(est)

    melhor = np.argmax([scores_ts])

    return {"train_esc": scores_tr, "test_mtr": scores_ts, "estimators": estfit, "best_idx": melhor}


def minha_cross_val(estimador, dados, predictors, target, n_splits=5, metrica=roc_auc_score):
    X, y = dados[predictors], dados[target]

    my_scorer = make_scorer(metrica, greater_is_better=True)

    search = cross_validate(estimador,
                            X, y,
                            scoring=my_scorer,
                            cv=n_splits,
                            return_estimator=True,
                            n_jobs=-1,
                            return_train_score=True)

    melhor = np.argmax(search['test_score'])

    return {"train_esc": search['train_score'], "test_esc": search['test_score'], "estimators": search['estimator'], "best_idx": melhor}


def retorna_correlations(dados, alpha=1):
    cls = dados.columns
    res = []
    preds = []
    for i, cl1 in enumerate(cls):
        for j, cl2 in enumerate(cls):
            cor = np.corrcoef(dados[cl1], dados[cl2])[0, 1]
            if j > i and np.abs(cor) <= alpha:
                res.append((cl1, cl2, cor))
                preds.append(cl1)
    return {"correlations": res, "a_predictors": list(set(preds))}

# Extract and sort feature coefficients


def get_feature_coefs(regression_model, label_index, columns):
    coef_dict = {}
    for coef, feat in zip(regression_model.coef_[label_index, :], columns):
        if abs(coef) >= 0.01:
            coef_dict[feat] = coef
    # Sort coefficients
    coef_dict = {k: v for k, v in sorted(
        coef_dict.items(), key=lambda item: item[1])}
    return coef_dict

# Generate bar colors based on if value is negative or positive


def get_bar_colors(values):
    color_vals = []
    for val in values:
        if val <= 0:
            color_vals.append('r')
        else:
            color_vals.append('g')
    return color_vals

# Visualize coefficients


def visualize_coefs(coef_dict):
    features = list(coef_dict.keys())
    values = list(coef_dict.values())
    y_pos = np.arange(len(features))
    color_vals = get_bar_colors(values)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y_pos, values, align='center', color=color_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    # labels read top-to-bottom
    ax.invert_yaxis()
    ax.set_xlabel('Feature Coefficients')
    ax.set_title('')
    plt.show()
