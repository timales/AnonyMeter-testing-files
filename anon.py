import pandas as pd
from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
import logging

logging.getLogger("anonymeter").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

ori = pd.read_csv('data/streaming_data.csv') 
syn = pd.read_csv('data/synthetic_data.csv') 
control = pd.read_csv('data/control_data.csv')

n_attacks = 100

ori.columns = [str(i+1) for i in range(ori.shape[1])]
syn.columns = [str(i+1) for i in range(syn.shape[1])]
control.columns = [str(i+1) for i in range(control.shape[1])]

print("Original Dataset:")
print(ori.info())
print("\nSynthetic Dataset:")
print(syn.info())
print("\nControl Dataset:")
print(control.info())

singling_out_evaluator = SinglingOutEvaluator(ori=ori, syn=syn, control=control, n_attacks=n_attacks)
# linkability_evaluator = LinkabilityEvaluator(ori=ori, syn=syn, control=control, n_attacks=n_attacks)
# inference_evaluator = InferenceEvaluator(ori=ori, syn=syn, control=control, n_attacks=n_attacks)

singling_out_evaluator.evaluate()
singling_out_risk = singling_out_evaluator.risk()
print(f"Singling-Out Score: {singling_out_risk}")
