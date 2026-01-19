import json
import re
from typing import Dict, List, Any
from flashrag.utils import get_retriever
from flashrag.config import Config
import os
import argparse

def remove_information_tags(text: str) -> str:
    """
    删除字符串中所有由<information>和</information>包裹的内容（包括标签本身）
    
    Args:
        text: 包含<information>标签的字符串
        
    Returns:
        删除所有<information>...</information>内容后的字符串
    """
    # 使用正则表达式匹配并删除<information>...</information>及其内容
    # re.DOTALL 让 . 匹配包括换行符在内的所有字符
    cleaned_text = re.sub(r'\n\n<information>.*?</information>\n\n', '', text, flags=re.DOTALL)
        
    return cleaned_text

def extract_llm_output(prompt: str, question: str) -> str:
    """从prompt中提取LLM输出部分"""
    # 构建完整的user_prompt
    # user_prompt = (
    #     "Answer the given question. "
    #     "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    #     "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. "
    #     "You can search as many times as your want. "
    #     "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
    #     f"Question: {question}\n"
    # )
    
    # 找到user_prompt的结束位置
    # user_prompt_end = prompt.find(user_prompt)
    # if user_prompt_end == -1:
    #     print("parse error!")
    
    # # LLM输出从user_prompt结束后开始
    # llm_start = user_prompt_end + len(user_prompt)
    # llm_output = prompt[llm_start:]

    llm_output = prompt.split('<|im_start|>assistant\n')[-1]
    return llm_output

def parse_search_rounds(data_item, llm_output: str, query_passageids) -> List[Dict[str, str]]:
    rounds = []
    # find all the positions of </search>
    search_end_positions = []
    for match in re.finditer(r'</search>', llm_output):
        search_end_positions.append(match.end())

    if not search_end_positions or len(search_end_positions) != len(data_item['output']['retrieval_results']):
        return rounds
    
    # deal with each round
    for i, search_end in enumerate(search_end_positions):
        current_round_end = search_end
        
        reasoning_context = llm_output[:current_round_end].strip()
        reasoning_context_without_docs = remove_information_tags(reasoning_context)
        current_query = data_item['output']['retrieval_results'][str(i)]['query']
        
        rounds.append({
            "round_index": i + 1,
            "reasoning_context": reasoning_context,
            # "reasoning_context_without_docs": reasoning_context_without_docs,
            "current_query": current_query,
            "candidate_passage_ids": query_passageids[data_item['id']][str(i)],
        })
    
    return rounds

def process_json_file(input_file: str, output_file: str, retriever, agentic_retriever_input=False):
    print(f'agentic_retriever_input={agentic_retriever_input}')
    # 读取输入文件
    with open(input_file, 'r') as f:
        data = json.load(f)

    all_search_queries = []
    for data_item in data:
        question = data_item['question']
        # history_queries = []
        for idx, turn_info in data_item['output']['retrieval_results'].items():
            if agentic_retriever_input:
                all_search_queries.append(turn_info['query_for_retrieval'])
            else:
                all_search_queries.append(turn_info['query'])
                
    _, passageids_list = retriever.batch_search_return_ids(all_search_queries)
    query_passageids = {}
    count = 0
    for data_item in data:
        query_passageids[data_item['id']] = {}
        for idx, turn_info in data_item['output']['retrieval_results'].items():
            query_passageids[data_item['id']][idx] = passageids_list[count]
            count += 1
    assert count == len(passageids_list)

    # query_passageids = {query: passageids for query, passageids in zip(all_search_queries, passageids_list)}
    processed_data = []
    print('start to postprocess')
    for i, data_item in enumerate(data):
        llm_output = extract_llm_output(data_item["output"]["prompt"], data_item["question"])
        trajectory = parse_search_rounds(data_item, llm_output, query_passageids)
        if trajectory == []:
            continue
        result = {
            "id": data_item["id"],
            "question": data_item["question"],
            "golden_answers": data_item["golden_answers"],
            "trajectory": trajectory,
            "output": {key: value for key, value in data_item['output'].items() if key != 'retrieval_results'}
        }
        processed_data.append(result)
    
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)
    print(f"Processing completed. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process trajectory data with configurable parameters')
    parser.add_argument('--generator_model_name', type=str, required=True,
                       help='Name of the generator model')
    parser.add_argument('--retrieval_model_path', type=str, required=True,
                       help='Path to the retrieval model')
    parser.add_argument('--index_path', type=str, required=True,
                       help='Path to the index')
    parser.add_argument('--agentic_retriever_input', type=bool, default=True,
                       help='Whether to use agentic retriever input')
    parser.add_argument('--dataset_names', type=str, nargs='+', required=True,
                       help='List of dataset names to process')
    
    args = parser.parse_args()

    for dataset_name in args.dataset_names:
        os.makedirs(f'training_data/trajectory/{args.generator_model_name}', exist_ok=True)
        os.makedirs(f'training_data/passage_candidates/{args.generator_model_name}', exist_ok=True)
        config_dict = {
            "gpu_id": "0,1,2,3,4,5,6,7",
            'framework': 'vllm',
            'generator_max_input_len': 16384,
            'generation_params': {'max_tokens': 512, 'skip_special_tokens': False},
            'generator_model_path':f'/root/paddlejob/workspace/trained_models/{args.generator_model_name}',
            'retrieval_model_path': args.retrieval_model_path,
            'index_path': args.index_path,
            'retrieval_topk': 20,
            'save_intermediate_data': False
        }
        config = Config("generate_trajectory_config.yaml", config_dict)
        retriever = get_retriever(config)
        print('retriever loaded')

        os.makedirs(f'training_data/passage_candidates/{args.generator_model_name}', exist_ok=True)
        process_json_file(
                        #   input_file=f'output/bamboogle_2025_10_05_11_58_search-r1/intermediate_data.json',
                        input_file=f'training_data/trajectory/{args.generator_model_name}/{dataset_name}.json',
                        #   output_file=f'training_data/passage_candidates/bamboogle.json',
                        output_file=f'training_data/passage_candidates/{args.generator_model_name}/{dataset_name}.json',
                        retriever=retriever,
                        agentic_retriever_input=args.agentic_retriever_input)