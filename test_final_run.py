import json
import random
import numpy as np
from train_genetic import evaluate_fitness  

random.seed(42)
np.random.seed(42)

#Best Weights
with open("best_weights.json") as f:
    best_weights_list = json.load(f)

best_weights1 = best_weights_list[0]
best_weights2 = best_weights_list[1]

final_score1 = evaluate_fitness(best_weights1, iterations=600)
final_score2 = evaluate_fitness(best_weights2, iterations=600)

print(f"Final test run score with weights1: {final_score1}")
print(f"Final test run score with weights2: {final_score2}")

with open("final_test_score.txt", "w") as f:
    f.write(f"Final test run score: {final_score1}\n")
    f.write(f"Final test run score: {final_score2}\n")
