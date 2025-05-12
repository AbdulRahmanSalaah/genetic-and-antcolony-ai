import json
import random
import numpy as np
from train_genetic import evaluate_fitness  

random.seed(42)
np.random.seed(42)

#Best Weights
with open("best_weights.json") as f:
    best_weights = json.load(f)


final_score = evaluate_fitness(best_weights, iterations=600)
print(f"Final test run score: {final_score}")

with open("final_test_score.txt", "w") as f:
    f.write(f"Final test run score: {final_score}\n")
