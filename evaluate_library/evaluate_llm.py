# Load the metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


# Obtain a description of each metric
print(accuracy.description)
print(precision.description)
print(recall.description)
print(f1.description)

# See the required data types
print(f"The required data types for accuracy are: {accuracy.features}.")
print(f"The required data types for precision are: {precision.features}.")
print(f"The required data types for recall are: {recall.features}.")
print(f"The required data types for f1 are: {f1.features}.")



accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

# Extract the new predictions
predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

# Compute the metrics by comparing real and predicted labels
print(accuracy.compute(references=validate_labels, predictions=predicted_labels))
print(precision.compute(references=validate_labels, predictions=predicted_labels))
print(recall.compute(references=validate_labels, predictions=predicted_labels))
print(f1.compute(references=validate_labels, predictions=predicted_labels))