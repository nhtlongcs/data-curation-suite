import os 
for dirpath, dirnames, filenames in os.walk(os.path.dirname(__file__)):
	for filename in filenames:
		print(os.path.abspath(os.path.join(dirpath, filename)))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from .vllm_setup import get_vllm_model, vllm_call
from .config import loadConfig
CONFIG = loadConfig()

class MyModel:
	def __init__(self):
		
		self.models = [get_vllm_model(
			config=CONFIG, i=i
		) for i in range(len(CONFIG.MODEL_LIST))]
		print("VLLM model loaded successfully.")

	def __call__(self, questions):
		print(f"Received {len(questions)} questions for inference.")
		print(questions[:10])

		# ALso by lexical order to better caching
		results = vllm_call(
			model=self.models[0],  # Assuming single model for simplicity
			questions=questions,
			config=CONFIG
		)

		# Make sure results only have questionID and answer
		# results = [
		# 	{'questionID': res['questionID'], 'answer': str(res['answer'])}
		# 	for res in results
		# ]

		# print(results)
		return results
	

def loadPipeline():
	return MyModel()


if __name__ == '__main__':

	pipeline = loadPipeline()

	questions = [{'questionID': 123, 'question': 'what is the capital of Ireland?'}, 
				 {'questionID': 456, 'question': 'what is the capital of Italy?'}]

	answers = pipeline(questions)

	print(answers)