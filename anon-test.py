import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from anonymeter.evaluators import SinglingOutEvaluator
import logging

logging.getLogger("anonymeter").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

iris = load_iris()
ori = pd.DataFrame(iris.data, columns=iris.feature_names)

ori, syn = train_test_split(ori, test_size=0.5, random_state=42)
syn, control = train_test_split(syn, test_size=0.5, random_state=42)

n_attacks = 1

singling_out_evaluator = SinglingOutEvaluator(ori=ori, syn=syn, control=control, n_attacks=n_attacks)

singling_out_evaluator.evaluate()
singling_out_risk = singling_out_evaluator.risk()
print(f"Singling-Out Score: {singling_out_risk}")
