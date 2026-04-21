# Calculate the individual toxicities
toxicity_1 = toxicity_metric.compute(predictions=user_1)
toxicity_2 = toxicity_metric.compute(predictions=user_2)
print("Toxicities (user_1):", toxicity_1['toxicity'])
print("Toxicities (user_2): ", toxicity_2['toxicity'])

# Calculate the maximum toxicities
toxicity_1_max = toxicity_metric.compute(predictions=user_1, aggregation="maximum")
toxicity_2_max = toxicity_metric.compute(predictions=user_2, aggregation="maximum")
print("Maximum toxicity (user_1):", toxicity_1_max['max_toxicity'])
print("Maximum toxicity (user_2): ", toxicity_2_max['max_toxicity'])

# Calculate the toxicity ratios
toxicity_1_ratio = toxicity_metric.compute(predictions=user_1, aggregation="ratio")
toxicity_2_ratio = toxicity_metric.compute(predictions=user_2, aggregation="ratio")
print("Toxicity ratio (user_1):", toxicity_1_ratio['toxicity_ratio'])
print("Toxicity ratio (user_2): ", toxicity_2_ratio['toxicity_ratio'])

# Load the regard and regard-comparison metrics
regard = evaluate.load("regard")
regard_comp = evaluate.load("regard", "compare")

# Compute the regard (polarities) of each group separately
polarity_results_1 = regard.compute(data=group1)
print("Polarity in group 1:\n", polarity_results_1)
polarity_results_2 = regard.compute(data=group2)
print("Polarity in group 2:\n", polarity_results_2)

# Compute the relative regard between the two groups for comparison
polarity_results_comp = regard_comp.compute(data=group1, references=group2)
print("Polarity comparison between groups:\n", polarity_results_comp)