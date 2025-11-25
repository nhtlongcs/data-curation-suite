import json
import inspect
import os

class ExperimentConfig:
    """Experiment Configuration"""
    
    # Directory Configuration
    # MODEL_CACHE_DIR = '/home/nhtlong/fast-llm-llamafile/cache-hf'
    MODEL_CACHE_DIR = "/app/models/"
    SRC_CACHE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))

    
    # Benchmark Configuration
    # Available datasets:
    datasets = [
        os.path.join(SRC_CACHE_DIR, "data/final/questions_1611.csv"),
        os.path.join(SRC_CACHE_DIR, "data/final/chinese.csv"),
        # "/app/src/data/final/samples/questions_100s_2211_42.csv"
        # "/app/src/data/final/pre_final_benchmark/sampled_chinese_data.csv"
        # "/app/src/data/final/pre_final_benchmark/sampled_math_data.csv",
        # "/app/src/data/final/pre_final_benchmark/sampled_hist_data.csv", 
        # "/app/src/data/final/pre_final_benchmark/sampled_geo_data.csv"
    ]
    
    # Inference Configuration
    USE_KV_CACHE = True
    SORT_QUESTIONS = True

    MODEL_LIST = [
        {
            "MODEL_NAME": "Qwen/Qwen3-0.6B",
            "GGUF_FILE": os.path.join(SRC_CACHE_DIR, 'inferencePipeline', "qwen3-0.6b.Q4_K_M.gguf"),
                                    #   "Qwen3-4B-Q4_K_M.gguf"),
        },
    ]
       
    # Routing Configuration
    # If number of question tokens exceed MAX_QUESTION_TOKENS, return DEFAULT_ANSWER
    
    class Routers:
        # Number of workers for concurrent requests
        N_WORKERS = 16
        BATCH_SIZE = 128
        SHUFFLE = False

        class Default:
            SYSTEM_PROMPT = "You are ChatGPT, an AI assistant. Answer as concisely as possible."
            PROMPT = "{question}"
            DEFAULT_ANSWER = ""  
            MAX_NEW_TOKENS = 50
        
        class TopicRouting(Default):
            SYSTEM_PROMPT = "You are a helpful assistant that classifies questions into topics: math, geography, history, chinese, or other."
            PROMPT = "Is this question related to math, geography, history, chinese or else? Answer with one word. Question: {question}"
            PORTS = [1234]

        class Math:
            SYSTEM_PROMPT = "Solve the following math problem step by step. Make sure to put the answer (and only answer) inside \\boxed{}.\n"
            # SYSTEM_PROMPT = "Solve the following math problem step by step. Make sure to put the answer (and only answer) inside \\boxed{}.\n"
            PROMPT = "\n{question}"
            DEFAULT_ANSWER = ""
            MAX_NEW_TOKENS = 512

            USE_PYTHON_CODE = False
            MAX_NEW_TOKENS_PYTHON = 512
            PYTHON_SYSTEM_PROMPT = "You excel at math and coding. Please provide the python code to solve the math problem by solving it step by step and returning the final answer. Import necessary Python libraries (only math, numpy and sympy)."
            PYTHON_PROMPT = "Solve this math question using Python code: {question}"

        class Geography:
            SYSTEM_PROMPT = "You are Geography Expert. Answer as concisely as possible."
            PROMPT = "{question}"
            DEFAULT_ANSWER = ""  
            MAX_NEW_TOKENS = 50

        class History:
            SYSTEM_PROMPT = "You are History Expert. Answer as concisely as possible."
            PROMPT = "{question}"
            DEFAULT_ANSWER = ""  
            MAX_NEW_TOKENS = 50

        class Chinese:
            # Chinese configuration placeholder
            SYSTEM_PROMPT = "你是一个知识问答助手，请直接回答问题。"
            PROMPT = "{question}"
            DEFAULT_ANSWER = ""  
            MAX_NEW_TOKENS = 50

    @staticmethod
    def get_topic_router_config(topic: str):
        topic = topic.lower()
        if topic == 'math' or topic == 'algebra':
            return ExperimentConfig.Routers.Math
        elif topic == 'geography':
            return ExperimentConfig.Routers.Geography
        elif topic == 'history':
            return ExperimentConfig.Routers.History
        elif topic == 'chinese':
            return ExperimentConfig.Routers.Chinese
        else:
            return ExperimentConfig.Routers.Default
    

def loadConfig():
    return ExperimentConfig()

def dump_config_instance(obj):
    """Dump an instance of ExperimentConfig into JSON-serializable dict."""
    cls = obj.__class__
    return class_to_dict(cls)

def class_to_dict(cls):
    """Convert class namespace (including nested classes) to a dict."""
    result = {}
    
    for name, value in vars(cls).items():
        if name.startswith("_"):
            continue
        if inspect.isroutine(value):
            continue

        # Nested class
        if inspect.isclass(value):
            result[name] = class_to_dict(value)
        else:
            result[name] = convert_value(value)

    return result

def convert_value(v):
    """Convert a single value into JSON-friendly types."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {k: convert_value(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [convert_value(x) for x in v]
    # fallback — stringify unsupported objects
    return str(v)

def save_config_to_file(config, filename):
    """Save the configuration to a JSON file."""
    config_dict = dump_config_instance(config)
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=4)

if __name__ == "__main__":

    config = ExperimentConfig()
    print("Model Cache Directory:", config.MODEL_CACHE_DIR)
    print("Using KV Cache:", config.USE_KV_CACHE)
    print("Math System Prompt:", config.Routers.Math.SYSTEM_PROMPT)
    save_config_to_file(config, "experiment_config.json")