import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#import torch
import json
import re
import os
import string
import time

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

#citations的bad case处理
# def move_citations_first(sents):
#     new_sents = []
#     for i, sent in enumerate(sents):
#          # 删除空括号 [ ]
#         sent = re.sub(r'\[\s*\]', '', sent)
#         # 场景3：处理特殊格式的引用
#         sent = re.sub(r'(\[\d+\]\s*,\s*)+', lambda m: re.sub(r'\s*,\s*', '', m.group(0)), sent)
        
#         # 场景1：当前句子仅包含引用标记
#         if re.fullmatch(r'(\s*\[\d+\]\s*)+', sent.strip()):
#             if i > 0:  # 确保不是第一个句子
#                 # 在前一个句子的句号前添加标记
#                 prev_sent = new_sents[-1]
#                 if '.' in prev_sent:
#                     # 替换最后一个句号为"引用标记."
#                     parts = prev_sent.rsplit('.', 1)
#                     new_prev_sent = parts[0].strip() + sent.strip() + '.' + parts[1].lstrip()
#                     new_sents[-1] = new_prev_sent
#                 else:
#                     # 如果没有句号，直接追加到末尾
#                     new_sents[-1] = prev_sent + ' ' + sent.strip()
#             continue
                
#         # 场景2：当前句子以引用标记开头
#         match = re.match(r'^(\s*(\[\d+\]\s*)+)', sent)
#         if match:
#             citations = match.group(1).strip()
#             remaining_text = sent[match.end():].strip()
            
#             if i > 0:  # 确保不是第一个句子
#                 # 将引用标记移动到前一个句子的句号前
#                 prev_sent = new_sents[-1]
#                 if '.' in prev_sent:
#                     parts = prev_sent.rsplit('.', 1)
#                     new_prev_sent = parts[0] + citations + '.' + parts[1]
#                     new_sents[-1] = new_prev_sent
#                 else:
#                     new_sents[-1] = prev_sent + ' ' + citations
#                 # 将剩余文本作为新句子
#                 if remaining_text:
#                     if remaining_text==".":
#                         continue
#                     new_sents.append(remaining_text)
#             else:
#                 # 如果是第一个句子，保留剩余文本
#                 if remaining_text:
#                     new_sents.append(remaining_text)
#             continue
        
#         # 如果句子只包含一个句号或空字符串，则不保留
#         if sent.strip() in ('.', ''):
#             continue
        
#         new_sents.append(sent)
#     return new_sents
# #引证去重
# def remove_duplicate_citations(sentence_sequence):
#     result = []
#     for sentence in sentence_sequence:
#         # 去掉引证之间的空格
#         sentence = re.sub(r'(\[\d+\])\s+(\[\d+\])', r'\1\2', sentence)
#         # 使用正则表达式查找引证部分
#         citations = re.findall(r'\[\d+\]', sentence)
#         citation_indices = {}
#         current_position = 0
#         for cit in citations:
#             # 查找当前引证的位置
#             index = sentence.find(cit, current_position)
#             if index != -1:
#                 citation_indices[cit] = index
#                 current_position = index + len(cit)
        
#         unique_citations = []
#         seen_citations = set()
#         for cit in citations:
#             if cit not in seen_citations:
#                 seen_citations.add(cit)
#                 unique_citations.append(cit)
        
#         # 按照在原文中出现的顺序排序
#         unique_citations_sorted = sorted(unique_citations, key=lambda x: citation_indices[x])
        
#         # 将去重后的引证替换回句子中
#         result_sentence = re.sub(r'\[\d+\]', '', sentence)
#         prev_pos = 0
#         final_sentence = ''
#         for cit in unique_citations_sorted:
#             pos = citation_indices[cit]
#             final_sentence += result_sentence[prev_pos:pos]
#             final_sentence += cit
#             prev_pos = pos + len(cit)
#         final_sentence += result_sentence[prev_pos:]
#         # 确保句子以句号结尾
#         if not re.search(r'\.$', final_sentence):
#             final_sentence += '.'
        
#         result.append(final_sentence)
#     return result

# #citations处理的最终调用       
# def move_citations(unprocessed_sents):
#     sents= move_citations_first(unprocessed_sents)
#     result = remove_duplicate_citations(sents)
#     return result


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


def make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=None):
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    text = doc['text']
    if use_shorter is not None:
        text = doc[use_shorter]
    return doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))


def get_shorter_text(item, docs, ndoc, key):
    doc_list = []
    for item_id, item in enumerate(docs):
        if key not in item:
            if len(doc_list) == 0:
                # If there aren't any document, at least provide one (using full text)
                item[key] = item['text']
                doc_list.append(item)
            logger.warn(f"No {key} found in document. It could be this data do not contain {key} or previous documents are not relevant. This is document {item_id}. This question will only have {len(doc_list)} documents.")
            break
        if "irrelevant" in item[key] or "Irrelevant" in item[key]:
            continue
        doc_list.append(item)
        if len(doc_list) >= ndoc:
            break
    return doc_list


def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, use_shorter=None, test=False):
    # For demo prompt
    # - {INST}: the instruction
    # - {D}: the documents
    # - {Q}: the question
    # - {A}: the answers
    # ndoc: number of documents to put in context
    # use_shorter: None, "summary", or "extraction"

    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "") # if there is no doc we also delete the empty line
        else:
            doc_list = get_shorter_text(item, item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n

    return prompt


# def load_model(model_name_or_path, dtype=torch.float16, int8=False, reserve_memory=10):
#     # Load a huggingface model and tokenizer
#     # dtype: torch.float16 or torch.bfloat16
#     # int8: whether to use int8 quantization
#     # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

#     # Load the FP16 model
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     logger.info(f"Loading {model_name_or_path} in {dtype}...")
#     if int8:
#         logger.warn("Use LLM.int8")
#     start_time = time.time()
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name_or_path,
#         device_map='auto',
#         torch_dtype=dtype,
#         max_memory=get_max_memory(),
#         load_in_8bit=int8,
#     )
#     logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

#     # Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

#     # Fix OPT bos token problem in HF
#     if "opt" in model_name_or_path:
#         tokenizer.bos_token = "<s>"
#     tokenizer.padding_side = "left"

#     return model, tokenizer
