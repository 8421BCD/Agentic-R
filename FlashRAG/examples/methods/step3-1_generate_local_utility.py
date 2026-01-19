import argparse
import os
from vllm import LLM, SamplingParams
import json
import re
from tqdm import tqdm
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# **Deep Search Process:**
# The Deep Search is a multi-turn process where the LLM solves the original user question by iteratively thinking and searching. Each turn involves:
# 1. The LLM's internal Think Content (the reasoning process to solve the original user question), wrapped in <think> and </think>.
# 2. The Search Query to supplement the missing knowledge, wrapped in <search> and </search>.
# 3. The search results (Passages), wrapped in <information> and </information>.

def construct_prompt(query, passages, tokenizer, use_subanswer=True, subanswer=None):
    if use_subanswer and subanswer is not None and subanswer.lower() != "not sure":
        sub_prompt1 = "I will also provide you with a reference answer to the Search Query to help you assess the utility score of each passage."
        sub_prompt2 = f"**Reference Answer:**\n{subanswer}"
    else:
        sub_prompt1, sub_prompt2 = '', ''
    system_message = "You are an expert evaluator whose role is to determine how effectively a passage answers a search query." 
    prompt_template = (
    """
Given a Search Query and a list of passages, you need to assign a utility score (from a scale of 0 to 100) for each passage. The utility score should reflect how well each passage **directly answers** or **contains the necessary information** to resolve the information need of the Search Query.
{sub_prompt1}

The scoring criteria is shown as follows:

* **81-100 (Excellent Utility):** The passage directly and comprehensively addresses the Search Query or completely contains the answer required by the current Search Query.
* **61-80 (High Utility):** The passage contains the majority of the information needed to directly answer the Search Query but might miss some minor details.
* **41-60 (Moderate Utility):** The passage is on-topic and addresses a part of the query's intent, but it is not a comprehensive answer.
* **21-40 (Low Utility):** The passage mentions keywords from the query, but its main topic is different. It offers very limited value.
* **0-20 (Very Low Utility):** The passage is completely irrelevant to the Search Query. It contains no useful facts or is actively distracting.

Now, I will give you the Search Query and {num} passages (each indicated by a number identifier []). 
Please compare all passages globally before outputting the utility scores, ensuring relative scoring consistency and more precise utility determination for each passage.

**Search Query:**
{query}

**Passages:**
{passages}

{sub_prompt2}

Please output the utility scores of the passages (strictly as integers between 0 and 100) in the order of the passage identifiers, separating the scores with a single space. If there is only one passage, output a single number. Example output for 3 passages: "68 42 97". Only output the utility scores, without any words or explanations.
"""
    )
# Note that I will also provide the specific Search Intent behind the Search Query as auxiliary context to help you better understand the specific information needs of the Search Query, as the Search Query itself may sometimes be vague.
# **Search Intent:**
# {think}
    passages_contents = ''
    for idx, passage in enumerate(passages):
        passages_contents += f"Passage [{idx+1}]: {passage}\n"
    user_content = prompt_template.format(query=query, passages=passages_contents, num=len(passages), sub_prompt1=sub_prompt1, sub_prompt2=sub_prompt2)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def get_data_to_process(args):
    # load all training data
    with open(args.input_data_path, 'r') as f:
        all_data = json.load(f)
    # load processed training ids
    done_ids = set()
    if os.path.exists(args.result_path):
        with open(args.result_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                done_ids.add(data['id'])
    data_to_process = [data for data in all_data if data['id'] not in done_ids]
    return data_to_process

def get_prompts(args, batch_data, id_doc, tokenizer, id_subanswers, use_subanswer=True):
    batch_prompts = []
    for data in batch_data:
        for round_info in data['trajectory']:
            if 'round_index' not in round_info or 'reasoning_context' not in round_info or 'current_query' not in round_info:
                continue
            round_id = round_info['round_index']
            # reasoning_context = round_info['reasoning_context']
            # matches = re.findall(r'<think>(.*?)</think>', reasoning_context, re.DOTALL)
            # if matches:
            #     current_think = matches[-1]
            # else:
            #     continue
            current_query = round_info['current_query']
            candidate_passage_ids = round_info['candidate_passage_ids']
            if data['id'] in id_subanswers:
                assert current_query == id_subanswers[data['id']][round_id - 1]['query']
            for start_passage_idx in range(0, len(candidate_passage_ids), args.batch_passages_size):
                passage_contents = [id_doc[str(passage_id)] for passage_id in candidate_passage_ids[start_passage_idx: start_passage_idx + args.batch_passages_size]]
                subanswer = id_subanswers[data['id']][round_id - 1]['subanswer'] if data['id'] in id_subanswers else None
                prompt = construct_prompt(current_query, passage_contents, tokenizer, use_subanswer=use_subanswer, subanswer=subanswer)
                batch_prompts.append(prompt)
    return batch_prompts

def get_score(score_text):
    score = int(score_text)
    score = max(0, score)
    score = min(100, score)
    return score



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--num_gpus", type=int)
    parser.add_argument("--input_data_path", type=str)
    parser.add_argument("--subanswer_data_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--batch_passages_size", type=int, default=10)
    parser.add_argument("--use_subanswer", type=bool, default=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    llm = LLM(args.model_name_or_path, 
                download_dir=os.getenv("HF_HOME"), 
                # max_model_len=4096,
                gpu_memory_utilization=0.9, 
                enforce_eager=False, 
                tensor_parallel_size=args.num_gpus
                )
    tokenizer = llm.get_tokenizer()

    data_to_process = get_data_to_process(args)
    # load subanswer
    print('loading subanswers')
    id_subanswers = {}
    with open(args.subanswer_data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            id_subanswers[data['id']] = data['query_subanswer_list']
    # load corpus
    print('loading corpus...')
    id_doc = {}
    with open('/root/paddlejob/workspace/data/FlashRAG_Dataset/retrieval_corpus/wiki18_100w.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            id_doc[data['id']] = data['contents']
    print('corpus loaded')
    
    # process the data
    for start_idx in tqdm(range(0, len(data_to_process), args.batch_size)):
        print(f'----------------- processing {start_idx} to {start_idx + args.batch_size}, totally {len(data_to_process)}....--------------')
        batch_data = data_to_process[start_idx: start_idx + args.batch_size]
        batch_prompts = get_prompts(args, batch_data, id_doc, tokenizer, id_subanswers, args.use_subanswer)
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1000,
            min_tokens=1,
        )
        print(batch_prompts[0])
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=True,)
        scores_list = [output.outputs[0].text.strip().split(' ') for output in outputs]
        
        with open(args.result_path, 'a') as f:
            scores_list_idx = 0
            for data in batch_data:
                for round_info in data['trajectory']:
                    if 'round_index' not in round_info or 'reasoning_context' not in round_info or 'current_query' not in round_info:
                        continue
                    error_flag = 0
                    docid_score = {}
                    passage_ids = round_info['candidate_passage_ids']

                    for start_passage_idx in range(0, len(passage_ids), args.batch_passages_size):
                        try:
                            current_batch_passage_ids = passage_ids[start_passage_idx: start_passage_idx + args.batch_passages_size]
                            for i in range(len(current_batch_passage_ids)):
                                docid_score[current_batch_passage_ids[i]] = get_score(scores_list[scores_list_idx][i])
                            scores_list_idx += 1
                        except:
                            print('llm output error, ignore this round')
                            scores_list_idx += 1
                            error_flag = 1
                            continue
                    if error_flag == 0:
                        result = {
                            "id": data['id'],
                            "round_index": round_info['round_index'],
                            "docid_score": docid_score,
                        }
                        f.write(json.dumps(result) + '\n')

        if len(scores_list) != scores_list_idx:
            print('len(scores_list) != scores_list_idx')