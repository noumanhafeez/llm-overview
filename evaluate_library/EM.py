# Load the metric
exact_match = evaluate.load("exact_match")

predictions = ["It's a wonderful day", "I love dogs", "DataCamp has great AI courses", "Sunshine and flowers"]
references = ["What a wonderful day", "I love cats", "DataCamp has great AI courses", "Sunsets and flowers"]

# Compute the exact match and print the results
results = exact_match.compute(references=references, predictions=predictions)
print("EM results: ", results)