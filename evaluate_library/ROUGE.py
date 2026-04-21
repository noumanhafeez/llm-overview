# Load the rouge metric
rouge = evaluate.load("rouge")

predictions = ["""Pluto is a dwarf planet in our solar system, located in the Kuiper Belt beyond Neptune, and was formerly considered the ninth planet until its reclassification in 2006."""]
references = ["""Pluto is a dwarf planet in the solar system, located in the Kuiper Belt beyond Neptune, and was previously deemed as a planet until it was reclassified in 2006."""]

# Calculate the rouge scores between the predicted and reference summaries
results = rouge.compute(predictions=predictions, references=references)
print("ROUGE results: ", results)