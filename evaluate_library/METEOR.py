meteor = evaluate.load("meteor")

generated = ["The burrow stretched forward like a narrow corridor for a while, then plunged abruptly downward, so quickly that Alice had no chance to stop herself before she was tumbling into an extremely deep shaft."]
reference = ["The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."]

# Compute and print the METEOR score
results = meteor.compute(predictions=generated, references=reference)
print("Meteor: ", results['meteor'])