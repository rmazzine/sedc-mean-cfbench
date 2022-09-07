import sys

sys.path.append('../')

import time

import numpy as np
import pandas as pd
from cfbench.cfbench import BenchmarkCF, TOTAL_FACTUAL

import explainer
from benchmark.utils import timeout, TimeoutError

# Get initial and final index if provided
if len(sys.argv) == 3:
    initial_idx = sys.argv[1]
    final_idx = sys.argv[2]
else:
    initial_idx = 0
    final_idx = TOTAL_FACTUAL

# Create Benchmark Generator
benchmark_generator = BenchmarkCF(
    output_number=1,
    show_progress=True,
    disable_tf2=True,
    disable_gpu=True,
    initial_idx=int(initial_idx),
    final_idx=int(final_idx)).create_generator()

# The Benchmark loop
sedc_current_dataset = None
for benchmark_data in benchmark_generator:
    # Get factual array
    factual_array = benchmark_data['factual_oh']

    # Get train data
    train_data = benchmark_data['df_oh_train']

    # Get columns info
    columns = list(train_data.columns)[:-1]

    # Get factual row as pd.Series
    factual_row = pd.Series(benchmark_data['factual_oh'], index=columns)

    # Get Keras TensorFlow model
    model = benchmark_data['model']

    # Get Evaluator
    evaluator = benchmark_data['cf_evaluator']

    if sedc_current_dataset != benchmark_data['dsname']:
        default_values = train_data.drop(columns=['output']).mean(axis=0).to_numpy().astype(float)
        # Decision boundary is 0.5 for binary classification
        decision_boundary = 0.5
        prediction_factual = model.predict(np.array([factual_array]).astype(float))[0][0]
        def scoring_function(X):
            return model.predict(X) if prediction_factual >= 0.5 else 1 - model.predict(X)
        explain = explainer.Explainer(scoring_function, default_values)

        sedc_current_dataset = benchmark_data['dsname']

    @timeout(600)
    def generate_cf():
        try:
            # Create CF using SEDC's explainer and measure generation time
            start_generation_time = time.time()
            explanation = explain.explain(np.array([factual_array]).astype(float), decision_boundary)
            cf_generation_time = time.time() - start_generation_time

            # Get first CF
            if len(explanation[0]) > 0:
                replace_idxs = explanation[0][0]
                cf = [f if f_idx not in replace_idxs else
                      default_values[f_idx] for f_idx, f in enumerate(factual_array)]
            else:
                cf = factual_row.to_list()
        except Exception as e:
            print('Error generating CF')
            print(e)
            # In case the CF generation fails, return same as factual
            cf = factual_row.to_list()
            cf_generation_time = np.NaN

        # Evaluate CF
        evaluator(
            cf_out=cf,
            algorithm_name='sedc',
            cf_generation_time=cf_generation_time,
            save_results=True)

    try:
        generate_cf()
    except TimeoutError:
        print('Timeout generating CF')
        # If CF generation time exceeded the limit
        evaluator(
            cf_out=factual_row.to_list(),
            algorithm_name='dice',
            cf_generation_time=np.NaN,
            save_results=True)
