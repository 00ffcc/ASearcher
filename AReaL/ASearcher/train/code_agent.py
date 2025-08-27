import queue
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

@dataclass
class Record:
    type: str # prompt/llm_gen/exec_results
    text: str
    # for exec_results
    short_text: str = ""
    # RL data
    input_len: Optional[int] = None
    input_tokens: Optional[List[int]] = None
    output_len: Optional[int] = None
    output_tokens: Optional[List[int]] = None
    output_logprobs: Optional[List[float]] = None
    output_versions: Optional[List[int]] = None

    def to_dict(self):
        return asdict(self)

class AgentMemory:
    def __init__(self, prompt):
        self.memory = [Record(type="prompt", text=prompt)]
    
    def llm_gen_count(self):
        return sum([r.type == "llm_gen" for r in self.memory])
    
    def filter_records(self, record_type):
        return [r for r in self.memory if r.type == record_type]
    
    def prepare_prompt(self):
        prompt = ""
        for r in self.memory:
            if r.type == "prompt":
                prompt = r.text
            elif r.type in ["exec_results"]:
                prompt = prompt + r.short_text
            elif r.type == "llm_gen":
                prompt = prompt + r.text
            else:
                raise RuntimeError(f"Unknown record type: {r.type}")
        return prompt
    
    def prepare_prompt_token_ids(self, tokenizer):
        token_ids = []
        for r in self.memory:
            if r.type == "prompt":
                token_ids += tokenizer.encode(r.text, add_special_tokens=False)
            elif r.type in ["exec_results"]:
                token_ids += tokenizer.encode(r.short_text, add_special_tokens=False)
            elif r.type == "llm_gen":
                token_ids += r.output_tokens
            else:
                raise RuntimeError(f"Unknown record type: {r.type}")
        return token_ids
    
    def add_record(self, r: Record):
        self.memory.append(r)
    
    def logging_stats(self) -> Dict:
        llm_gens = self.filter_records(record_type="llm_gen")
        ret = dict(
            num_llm_gens=len(llm_gens),
            num_input_tokens=sum([len(r.input_tokens) for r in llm_gens]),
            num_output_tokens=sum([len(r.output_tokens) for r in llm_gens]),
        )
        return ret
    
    def to_dict(self):
        return [r.to_dict() for r in self.memory]

class CodeAgent:
    def __init__(self, prompt):
        self.prompt = prompt
        self.memory = AgentMemory(prompt=prompt)
        self.summary_job_queue = queue.Queue(128)
    
    @property
    def num_turns(self):
        return self.memory.llm_gen_count()
    
    @property
    def is_finished(self):
        pattern = r'<answer>(.*?)</answer>'
        return any([len(re.findall(pattern, r.text, re.DOTALL)) > 0 for r in self.memory.filter_records("llm_gen")])
    
    def add_summary_jobs(self, summary_jobs):
        if not isinstance(summary_jobs, list):
            summary_jobs = [summary_jobs]
        for summary_job in summary_jobs:
            self.summary_job_queue.put_nowait(summary_job)
    
    def prepare_llm_query(self):
        prompt = self.memory.prepare_prompt()
        sampling_params = dict(stop=["</python>", "</answer>"])
        if not self.summary_job_queue.empty():
            summary_job = self.summary_job_queue.get_nowait()
            if summary_job["type"] in ["exec_results"]:
                prompt = prompt + f"\n\n{summary_job['text']}\n<think>\n"
                new_record = Record(
                    type=summary_job["type"], 
                    text=f"\n\n{summary_job['text']}\n<think>\n", 
                    short_text=f"\n\n{summary_job['text']}\n<think>\n",
                )
                self.memory.add_record(new_record)
                sampling_params["stop"] = ["</think>"]
        return prompt, sampling_params
    
    def consume_llm_response(self, resp, completion_text):
        new_record = Record(
            type="llm_gen",
            text=completion_text,
            input_len=resp.input_len,
            input_tokens=resp.input_tokens,
            output_len=resp.output_len,
            output_tokens=resp.output_tokens,
            output_logprobs=resp.output_logprobs,
            output_versions=resp.output_versions            
        )
        self.memory.add_record(new_record)

        tool_calls = []
        for pattern in [r'<python>(.*?)</python>', r'<answer>(.*?)</answer>']:
            matches = re.findall(pattern, completion_text, re.DOTALL)
            if matches:
                match = matches[-1]
                tool_calls.append(str(pattern.replace('(.*?)', match)))

        return tool_calls

    def consume_tool_response(self, res):
        if res["type"] == "python":
            summary_job = dict(type="exec_results")

            stdout = res["response"]["run_result"]["stdout"]
            stderr = res["response"]["run_result"]["stderr"]
            status = res["response"]["run_result"]["status"]
            summary_job["text"] = f"<exec_results>\nStatus: {status}\nStdout:\n{stdout}\nStderr:\n{stderr}\n</exec_results>"
            self.add_summary_jobs(summary_job)


    def get_answer(self):
        text, _ = self.prepare_llm_query()
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None
