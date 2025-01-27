# Using real synthetic data to test extension
# Error: Found 100 failed queries out of 100. Check DEBUG messages for more details.

import pandas as pd
from anonymeter.evaluators import SinglingOutEvaluator
import logging

logging.getLogger("anonymeter").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

ori = pd.read_csv('data/streaming_data.csv') 
syn = pd.read_csv('data/synthetic_data.csv') 
control = pd.read_csv('data/control_data.csv')

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