# Translate the input sentences, extract the translated text, and compute BLEU score
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

translated_outputs = translator(input_sentences_2)

predictions = [translated_output['translation_text'] for translated_output in translated_outputs]
print(predictions)

results = bleu.compute(predictions=predictions, references=references_2)
print(results)