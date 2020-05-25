import re
import numpy as np
import pandas as pd


def eval_model_returning_only_text_level_metrics(manager, model, X_train, X_test, y_train, y_test, df_original):
    df_original["ID"] = df_original["ID"].astype(int)
    re_date = re.compile(r"[-\.]")
    df_original["date_accident"] = df_original["date_accident"].map(lambda x: re_date.sub("", str(x))).astype(str)
    df_original["date_consolidation"] = df_original["date_consolidation"].map(lambda x: re_date.sub("", str(x))).astype \
        (str)

    # Train model on X_train data
    if manager.is_fitted_:
        pass
    else:
        model = manager.fit_model(model, X_train, y_train)

    # Train
    df = manager.make_prediction(model, X_train)
    df = df.merge(df_original[["ID", "date_accident", "date_consolidation"]], on="ID")

    acc_acd_train1 = (df["date_accident_pred"] == df["date_accident"]).sum( ) /df.shape[0]
    df_tmp = df[~df["date_accident"].isin(["na", "nc", "n.a.", "n.c."])]
    acc_acd_train2 = (df_tmp["date_accident_pred"] == df_tmp["date_accident"]).sum() / df_tmp.shape[0]

    acc_conso_train1 = (df["date_consolidation_pred"] == df["date_consolidation"]).sum() / df.shape[0]
    df_tmp = df[~df["date_consolidation"].isin(["na", "nc", "n.a.", "n.c."])]
    acc_conso_train2 = (df_tmp["date_consolidation_pred"] == df_tmp["date_consolidation"]).sum() / df_tmp.shape[0]

    mean_train1 = np.mean([acc_acd_train1, acc_conso_train1])
    mean_train2 = np.mean([acc_acd_train2, acc_conso_train2])

    # Test
    df = manager.make_prediction(model, X_test)
    df = df.merge(df_original[["ID", "date_accident", "date_consolidation"]], on="ID")

    acc_acd_test1 = (df["date_accident_pred"] == df["date_accident"]).sum() / df.shape[0]
    df_tmp = df[~df["date_accident"].isin(["na", "nc", "n.a.", "n.c."])]
    acc_acd_test2 = (df_tmp["date_accident_pred"] == df_tmp["date_accident"]).sum() / df_tmp.shape[0]

    acc_conso_test1 = (df["date_consolidation_pred"] == df["date_consolidation"]).sum() / df.shape[0]
    df_tmp = df[~df["date_consolidation"].isin(["na", "nc", "n.a.", "n.c."])]
    acc_conso_test2 = (df_tmp["date_consolidation_pred"] == df_tmp["date_consolidation"]).sum() / df_tmp.shape[0]

    mean_test1 = np.mean([acc_acd_test1, acc_conso_test1])
    mean_test2 = np.mean([acc_acd_test2, acc_conso_test2])

    res = "Train :\n" \
          " Accuracy = %.2f (accident = %.2f, consolidation = %.2f)\n" \
          " Sans na/nc, accuracy = %.2f (accident = %.2f, consolidation = %.2f)\n" \
          "Test : \n" \
          " Accuracy = %.2f (accident = %.2f, consolidation = %.2f)\n" \
          " Sans na/nc, accuracy = %.2f (accident = %.2f, consolidation = %.2f)" % \
          (mean_train1, acc_acd_train1, acc_conso_train1, mean_train2, acc_acd_train2, acc_conso_train2,
           mean_test1, acc_acd_test1, acc_conso_test1, mean_test2, acc_acd_test2, acc_conso_test2)
    print(res)
    df_res = pd.DataFrame({"acc_moy" : [mean_train1, mean_train2, mean_test1, mean_test2],
                           "acc_accident" : [acc_acd_train1, acc_acd_train2, acc_acd_test1, acc_acd_test2],
                           "acc_consolidation" : [acc_conso_train1, acc_conso_train2, acc_conso_test1, acc_conso_test2]},
                          index=["train", "train_wo_nanc", "test", "test_wo_nanc"])
    return df_res

def intermediate_prediction_to_final_predictions(df):
    df["rank1"] = df.groupby("ID")["proba1"].rank("dense", ascending=False)
    df["rank2"] = df.groupby("ID")["proba2"].rank("dense", ascending=False)

    df1 = df[df["rank1"] == 1].copy().drop_duplicates(subset=["ID"])
    df1 = df1[["ID", "date_possible", "proba1"]]
    df1 = df1.rename(columns={"date_possible": "date_accident_pred",
                              "proba1": "proba_accident"})

    df2 = df[df["rank2"] == 1].copy().drop_duplicates(subset=["ID"])
    df2 = df2[["ID", "date_possible", "proba2"]]
    df2 = df2.rename(columns={"date_possible": "date_consolidation_pred",
                              "proba2": "proba_consolidation"})

    df_pred = df1.merge(df2, on="ID")
    return df_pred

def adjust_predictions_with_trhesholds(df_pred, th_nc_accident, th_na_consolidation, th_nc_consolidation):
    df_pred.loc[df_pred["proba_accident"] < th_nc_accident, "date_accident_pred"] = "nc"

    if th_nc_consolidation < th_na_consolidation:
        df_pred.loc[df_pred["proba_consolidation"] < th_nc_consolidation, "date_consolidation_pred"] = "nc"
        df_pred.loc[(df_pred["proba_consolidation"] > th_nc_consolidation) &
                    (df_pred["proba_consolidation"] < th_na_consolidation), "date_consolidation_pred"] = "na"
    else:
        df_pred.loc[df_pred["proba_consolidation"] < th_na_consolidation, "date_consolidation_pred"] = "na"
        df_pred.loc[(df_pred["proba_consolidation"] > th_na_consolidation) &
                    (df_pred["proba_consolidation"] < th_nc_consolidation), "date_consolidation_pred"] = "nc"

    return df_pred