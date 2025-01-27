# Simulating synthetic data to test extension
# Seems to perform without any problems

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
from anonymeter.evaluators import SinglingOutEvaluator
import logging

logging.getLogger("anonymeter").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

diabetes = load_diabetes()
ori = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

ori, syn = train_test_split(ori, test_size=0.5)
syn, control = train_test_split(syn, test_size=0.5)

evaluator = SinglingOutEvaluator(ori=ori, 
                                 syn=syn, 
                                 control=control,
                                 n_attacks=100)

try:
    evaluator.evaluate(mode='univariate')
    risk = evaluator.risk()
    print(risk)

except RuntimeError as ex: 
    print(f"Singling out evaluation failed with {ex}. Please re-run this cell."
          "For more stable results increase `n_attacks`. Note that this will "
          "make the evaluation slower.")