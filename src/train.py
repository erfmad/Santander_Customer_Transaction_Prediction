import argparse
import os

import config
import model_dispatcher

import joblib
import pandas as pd 
from sklearn import metrics
from sklearn import tree



def run(fold,model):
    df = pd.read_csv(config.TRAINING_FILE)

    for i in df.columns:
        if i not in ("target","ID_code"):
            df[f"bin_{i}"] = pd.cut(df[i],bins=10,labels=False)#in binning faghat rooie model lin moaser bood brdoone bin 0.9128 bood bade bin shod 0.91485

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)


    x_train = df_train.drop(["target","ID_code"],axis=1).values 
    y_train = df_train.target.values

    x_valid = df_valid.drop(["target","ID_code"],axis=1).values
    y_valid = df_valid.target.values 

    clf = model_dispatcher.models[model]

    clf.fit(x_train,y_train)

    preds = clf.predict(x_valid)

    accuracy = metrics.accuracy_score(y_valid,preds)
    print(f"Fold={fold},Accuracy={accuracy}")

    joblib.dump(clf,os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    args = parser.parse_args()

    run(
        fold= args.fold,
        model=args.model
    )

   
