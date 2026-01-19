import argparse
import os
import json
import re
import random
import numpy as np

if __name__ == "__main__":
    agent_name='triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em' # iter0
    # agent_name='triviaqa_hotpotqa_train-search-r1-ppo-qwen2.5-7b-em-iter1' # iter1
    datasets = ['hotpotqa', 'triviaqa']
    use_global_score=True
    use_local_score=True
    threshold = 0.6
    max_pos_num_per_query = 3
    training_passage_num = 16
    result_path = f'training_data/retriever_training_data.jsonl'

    result_data_list = []
    if use_global_score == False and use_local_score == False:
        print('either use_global_score or use_local_score must be True!')
        exit()
    for dataset in datasets:
        local_utility_path = f'training_data/local_utility/{agent_name}/{dataset}.jsonl'
        global_utility_path = f'training_data/global_utility/{agent_name}/{dataset}.jsonl'
        passage_candidates_path = f'training_data/passage_candidates/{agent_name}/{dataset}.json'
        
        id_passage_candidates = {}
        with open(passage_candidates_path, 'r') as f:
            data_list = json.load(f)
            for data in data_list:
                id_passage_candidates[data['id']] = data

        # load global score
        qid_roundid_info = {}
        if use_global_score:
            with open(global_utility_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    query_id = data['id']
                    question = data['question']
                    round_index = data['round_index']
                    history_queries = data['history_queries']
                    current_query = data['current_query']
                    docid_score = data['docid_score']
                    max_retrieved_times = max([info['retrieved_times'] for docid, info in docid_score.items()])
                    assert max_retrieved_times + round_index <= 5

                    if query_id not in qid_roundid_info:
                        qid_roundid_info[query_id] = {}
                    if round_index not in qid_roundid_info[query_id]:
                        qid_roundid_info[query_id][round_index] = {}
                    qid_roundid_info[query_id][round_index] = {
                        'dataset': dataset,
                        'query_id': query_id,
                        'question': question,
                        'round_index': round_index,
                        'history_queries': history_queries,
                        'query_text': current_query,
                        'docid_info': {docid: {
                                            'em': info['metric_score']['em'], 
                                            'retrieved_times': info['retrieved_times'],
                                            'success_retrieved_times': info['retrieved_times'] if info['metric_score']['em'] == 1 else 99999} 
                                        for docid, info in docid_score.items()},
                    }

        if use_local_score:
            with open(local_utility_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    query_id = data['id']
                    round_index = data['round_index']
                    query_text = id_passage_candidates[query_id]['trajectory'][round_index - 1]['current_query']
                    assert id_passage_candidates[query_id]['trajectory'][round_index - 1]['round_index'] == round_index

                    if query_id not in qid_roundid_info:
                        qid_roundid_info[query_id] = {}
                    if round_index not in qid_roundid_info[query_id]:
                        qid_roundid_info[query_id][round_index] = {
                            'dataset': dataset,
                            'query_id': query_id,
                            'round_index': round_index,
                            'query_text': query_text,
                            'docid_info': {docid: {'local_utility': local_utility / 100} 
                                            for docid, local_utility in data['docid_score'].items()},
                        }
                    else:
                        for docid, local_utility in data['docid_score'].items():
                            qid_roundid_info[query_id][round_index]['docid_info'][docid]['local_utility'] =  local_utility / 100
        count, count1 = 0, 0
        total = 0
        new_pos_num_list = []
        for qid, roundid_info in qid_roundid_info.items():
            for roundid, info in roundid_info.items():
                docid_info = info['docid_info']
                docid_info_list = [] # each item contains the docid and utility of each candidate passage
                for docid, utility_info in docid_info.items():
                    doc_info = {'docid': docid}
                    doc_info.update(utility_info)
                    docid_info_list.append(doc_info)


                positive_candidates, all_negative_candidates = [], []
                # sort
                if use_global_score and use_local_score: # to change
                    if 'local_utility' not in docid_info_list[0] or 'em' not in docid_info_list[0]:
                        continue
                    sorted_docid_info_list = sorted(docid_info_list, key=lambda x: (-x['em'], -x['local_utility']))
                    
                    # filter low quality positive
                    if sorted_docid_info_list[0]['local_utility'] < threshold or sorted_docid_info_list[0]['em'] == 0:  # to change
                        continue
                    first_candidate = sorted_docid_info_list[0]

                    for candidate in sorted_docid_info_list: # to change
                        if candidate['em'] == first_candidate['em'] and candidate['local_utility'] == first_candidate['local_utility']:
                            positive_candidates.append(candidate)
                        else:
                            all_negative_candidates.append(candidate)
                elif use_local_score:
                    sorted_docid_info_list = sorted(docid_info_list, key=lambda x: (-x['local_utility']))
                    if sorted_docid_info_list[0]['local_utility'] < threshold:
                        continue
                    first_candidate = sorted_docid_info_list[0]
                    for candidate in sorted_docid_info_list: # to change
                        if candidate['local_utility'] == first_candidate['local_utility']:
                            positive_candidates.append(candidate)
                        else:
                            all_negative_candidates.append(candidate)
                else:
                    sorted_docid_info_list = sorted(docid_info_list, key=lambda x: (-x['em']))
                    if sorted_docid_info_list[0]['em'] == 0:
                        continue
                    first_candidate = sorted_docid_info_list[0]
                    for candidate in sorted_docid_info_list: # to change
                        if candidate['em'] == first_candidate['em']:
                            positive_candidates.append(candidate)
                        else:
                            all_negative_candidates.append(candidate)
            
                if len(all_negative_candidates) < training_passage_num - 1:
                    continue
                idxs = sorted(random.sample(range(len(all_negative_candidates)), training_passage_num - 1))
                sampled_negative_candidates = [all_negative_candidates[idx] for idx in idxs]
                for positive_candidate in positive_candidates[:max_pos_num_per_query]:
                    all_candidates = [positive_candidate] + sampled_negative_candidates
                    passage_ids, scores = [], []
                    for candidate in all_candidates:
                        passage_ids.append(candidate['docid'])
                        scores.append({
                            'em': candidate.get('em', 0),
                            'local_utility': candidate.get('local_utility', 0),
                        })
                    result_data = {
                        'dataset': info['dataset'],
                        'query_id': info['query_id'],
                        'question': info.get('question', ''),
                        'round_index': info['round_index'],
                        'history_queries': info.get('history_queries', []),
                        'query_text': info['query_text'],
                        'passage_ids': passage_ids,
                        'scores': scores,
                    }
                    result_data_list.append(result_data)

    with open(result_path, 'w') as f:
        for data in result_data_list:
            f.write(json.dumps(data) + '\n')
