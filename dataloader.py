from typing import *
import torch.utils.data as data

from .config import ExperimentConfig

class CustomDataset(data.Dataset):
    """Custom Dataset for loading data from a CSV file."""
    def __init__(
            self, questions:List[dict], 
            tokenizer, sort_questions: bool = False,
            system_prompt: str = "",
            user_prompt: str = "{question}",
            no_think: bool = True,
        ):
        self.questions = questions
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.sort_questions = sort_questions
        self.no_think = no_think
        if self.sort_questions:
            self.questions = sorted(self.questions, key=lambda x: (len(x['question'].split()), x['question']))
    def __len__(self):
        return len(self.questions)
    def __getitem__(self, idx):
        question =  self.questions[idx]['question']

        if self.no_think:
            prompt = '/no_think ' + self.user_prompt.format(question=question)
        else:
            prompt = self.user_prompt.format(question=question)

        prompt_template = [
            {"role": "system", "content": self.system_prompt}, 
            {"role": "user", "content": prompt}
        ] 
        prompt_template = self.tokenizer.apply_chat_template(
            prompt_template, tokenize=False, add_generation_prompt=True
        )
        return {
            'question': question,
            'prompt_template': prompt_template,
            'questionID': self.questions[idx]['questionID'],
            'subject': self.questions[idx].get('subject', 'general'),
        }
    
    def collate_fn(self, batch):
        return batch
 

class CustomDataLoader(data.DataLoader):
    """Custom DataLoader for batching data from CustomDataset."""
    def __init__(self, dataset: CustomDataset, batch_size: int, shuffle: bool = True, num_workers: int = 0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    def __getitem__(self, index):
        return super().__getitem__(index)

def get_dataloader(questions: List[dict],  tokenizer, config: ExperimentConfig, subject: str = 'general', use_python_code: bool = False, no_think: bool = True) -> CustomDataLoader:
    """Create a DataLoader for the given dataset.
    Args:
        use_python_code (bool): To override the use of python code execution for math questions, otherwise use config.Routers.Math.USE_PYTHON_CODE
    """

    # Implement the logic to create and return a DataLoader
    router_config = ExperimentConfig.get_topic_router_config(subject)

    if subject in ['math', 'algebra'] and getattr(router_config, 'PYTHON_SYSTEM_PROMPT', False) and use_python_code:
        dataset = CustomDataset(
            questions=questions, 
            tokenizer=tokenizer, 
            sort_questions=config.SORT_QUESTIONS, 
            system_prompt=router_config.PYTHON_SYSTEM_PROMPT, 
            user_prompt=router_config.PYTHON_PROMPT,
            no_think=no_think,
        )
    else:
        dataset = CustomDataset(
            questions=questions, 
            tokenizer=tokenizer, 
            sort_questions=config.SORT_QUESTIONS, 
            system_prompt=router_config.SYSTEM_PROMPT, 
            user_prompt=router_config.PROMPT,
            no_think=no_think,
        )
    dataloader = CustomDataLoader(dataset, batch_size=config.Routers.BATCH_SIZE, shuffle=config.Routers.SHUFFLE, num_workers=config.Routers.N_WORKERS)
    return dataloader