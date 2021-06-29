from sklearn import tree
from sklearn import ensemble
import xgboost as xgb 
from sklearn import linear_model


models = {
    "tree": tree.DecisionTreeClassifier(),
    "rf": ensemble.RandomForestClassifier(),
    "xgb1": xgb.XGBClassifier(),
    "lin": linear_model.LogisticRegression()
   

}
