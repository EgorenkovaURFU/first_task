from transformer import pipeline

classifier = pipeline('sentiment-analysis',
		      'blanchefort/rubert-base-cased-sentiment')

classifier("I love running!")
