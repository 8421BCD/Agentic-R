from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers import PreTrainedModel, AutoModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers.file_utils import ModelOutput
from tevatron.retriever.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments
from torch.nn import BCELoss, BCEWithLogitsLoss
from itertools import product

import logging
logger = logging.getLogger(__name__)

def rank_net(y_pred, y_true=None, padded_value_indicator=-100, weight_by_diff=False,
                weight_by_diff_powed=False):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    if y_true is None:
        y_true = torch.zeros_like(y_pred).to(y_pred.device)
        y_true[:, 0] = 1

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)

@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 train_args = None,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.train_args = train_args

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, llm_scores = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            # 保存原始reps用于RankNet计算
            original_q_reps = q_reps
            original_p_reps = p_reps
            
            # 对比学习：使用gather后的全局tensor
            if self.is_ddp:
                q_reps_global = self._dist_gather_tensor(q_reps)
                p_reps_global = self._dist_gather_tensor(p_reps)
            else:
                q_reps_global = q_reps
                p_reps_global = p_reps

            # 对比学习loss
            scores = self.compute_similarity(q_reps_global, p_reps_global)
            scores = scores.view(q_reps_global.size(0), -1)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps_global.size(0) // q_reps_global.size(0))
            contrastive_loss = self.compute_loss(scores / self.temperature, target)
            
            if self.is_ddp:
                contrastive_loss = contrastive_loss * self.world_size

            if self.train_args.use_rankloss == False:
                final_loss = contrastive_loss
            else:
                original_scores = self.compute_similarity(original_q_reps, original_p_reps)
                # 提取每个query对应的passages的scores
                group_indices = self.select_grouped_indices(
                    scores=original_scores,
                    group_size=llm_scores.shape[1],
                )
                group_scores = torch.gather(input=original_scores, dim=1, index=group_indices)
                
                assert llm_scores.shape == group_scores.shape, \
                    f"llm_scores shape {llm_scores.shape} != scores shape {group_scores.shape}"
                
                batch_size, group_size = group_scores.shape
                rank_gt = 1 / torch.arange(1, 1 + group_size).view(1, -1).repeat(batch_size, 1).to(group_scores.device)
                
                for i in range(batch_size):
                    for j in range(1, group_size):
                        if llm_scores[i][j] == llm_scores[i][j - 1]:
                            rank_gt[i][j] = rank_gt[i][j - 1]
                
                rank_loss = rank_net(group_scores, rank_gt)
                final_loss = (1 - self.train_args.rankloss_weight) * contrastive_loss + self.train_args.rankloss_weight * rank_loss
    
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            final_loss = None
            
        return EncoderOutput(
            loss=final_loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    # # original forward version
    # def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
    #     q_reps = self.encode_query(query) if query else None
    #     p_reps = self.encode_passage(passage) if passage else None

    #     # for inference
    #     if q_reps is None or p_reps is None:
    #         return EncoderOutput(
    #             q_reps=q_reps,
    #             p_reps=p_reps
    #         )

    #     # for training
    #     if self.training:
    #         if self.is_ddp:
    #             q_reps = self._dist_gather_tensor(q_reps)
    #             p_reps = self._dist_gather_tensor(p_reps)

    #         scores = self.compute_similarity(q_reps, p_reps)
    #         scores = scores.view(q_reps.size(0), -1)

    #         target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
    #         target = target * (p_reps.size(0) // q_reps.size(0))

    #         loss = self.compute_loss(scores / self.temperature, target)
    #         if self.is_ddp:
    #             loss = loss * self.world_size  # counter average weight reduction
    #     # for eval
    #     else:
    #         scores = self.compute_similarity(q_reps, p_reps)
    #         loss = None
    #     return EncoderOutput(
    #         loss=loss,
    #         scores=scores,
    #         q_reps=q_reps,
    #         p_reps=p_reps,
    #     )

    def select_grouped_indices(self,
                            scores: torch.Tensor,
                            group_size: int,
                            start: int = 0) -> torch.Tensor:
        assert len(scores.shape) == 2
        batch_size = scores.shape[0]
        assert batch_size * group_size <= scores.shape[1]

        indices = torch.arange(0, group_size, dtype=torch.long)
        indices = indices.repeat(batch_size, 1)
        indices += torch.arange(0, batch_size, dtype=torch.long).unsqueeze(-1) * group_size
        indices += start
        return indices.to(scores.device)

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                train_args=train_args,
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                train_args=train_args,
            )
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize
            )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
        print(f'output_dir: {output_dir}')