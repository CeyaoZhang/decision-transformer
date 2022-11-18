'''
This is stem from the UniMASK 
https://github.com/micahcarroll/uniMASK/blob/main/uniMASK/batches.py
'''


from abc import ABC

import numpy as np
import torch
from torch import tensor as tt





class Batch(ABC):
    """
    A batch needs to have all the information necessary to create it's own component of the model input to the
    transformer:
    [ batch_len , num_tokens, max_token_size ]

    And also needs to keep track of what outputs it should attend

    We want to be able to calculate accuracy, loss, and other statistics by batch, as this will serve us for debugging
    purposes.
    """

    RTG_MASKING_TYPES = [
        "BC",
        "RC_fixed",
        "RC_fixed_first",
        "BCRC_uniform_all",
        "BCRC_uniform_only_zeros",
        "BCRC_uniform_first",
        "Unchanged",
    ]

    def __init__(
        self,
        # input_data,
        num_seqs, ## I add
        seq_len, ## I add
        device, ## I add
        rtg_masking_type="Unchanged",
        silent=False,
        **kwargs,
    ):
        # self.input_data = input_data
        # self.mask_shape = input_data.shape
        # self.seq_len = self.input_data.seq_len
        # self.num_seqs = self.input_data.num_seqs
        self.seq_len = seq_len ## I add
        self.num_seqs = num_seqs ## batch size 
        self.device = device ## I add

        self.rtg_masking_type = rtg_masking_type

        self.silent = silent

        # This has to be computed in the subclasses' init methods
        self.input_masks = None
        self.prediction_masks = None

        self.loss = None

        self.computed_output = False

    # @classmethod
    # def from_params(cls, input_data, batch_params):
    #     batch_class = batch_params["type"]
    #     return batch_class(input_data=input_data, **batch_params)

    def get_input_masks(self):
        """Get the input masks for observations and feeds. All logic will be in subclasses"""
        raise NotImplementedError()

    @staticmethod
    def postprocess_rtg_mask(rtg_mask, rtg_masking_type):
        """
        Various kinds of rtg masking.

        - BC: behavior cloning. Will mask all rtg info, always.
        - RC_fixed: reward-conditioning without randomization (all rtg tokens always present).
        - RC_fixed_first: reward-conditioning without randomization (first rtg tokens always present, rest always masked).
        - BCRC_uniform_all: with 50% chance masks all rtg info, and 50% _shows all of it_
        - BCRC_uniform_only_zeros: with 50% chance masks all rtg info, and 50% leaves mask the same as what the batch decided
        - BCRC_uniform_first: with 50% chance mask first rtg. Rest will always be masked
        - Unchanged: just keep whatever masking scheme the batch type generates
        """
        assert rtg_masking_type in Batch.RTG_MASKING_TYPES

        num_seqs, seq_len = rtg_mask.shape ## rtg_mask = (batch, length)
        if rtg_masking_type == "BC":
            rtg_mask[:, :] = 0

        elif rtg_masking_type == "RC_fixed":
            rtg_mask[:, :] = 1

        elif rtg_masking_type == "RC_fixed_first":
            rtg_mask[:, 0] = 1
            rtg_mask[:, 1:] = 0

        elif rtg_masking_type == "BCRC_uniform_all":
            # NOTE: think hard before using this. Sometimes this won't lead to the desired effects
            # Consider using FuturePred batch, with randomized masking. Sometimes you'll want to predict the
            # action 3 with state and rtg up to timestep 3.
            # Using BCRC_uniform in that case will never enable you to do that. When it selects that you should see
            # rtg, it will show you all of the trajectory's RTG, leaking information about the reward.
            # For that case you should use BCRC_uniform_zero_only which only zeros out RTGs (but doesn't reveal any more than the batch class has decided to)
            uniform_rtg_mask = np.random.choice([0, 1], size=[num_seqs])
            rtg_mask[:, :] = tt(np.vstack([uniform_rtg_mask] * seq_len)).T

        elif rtg_masking_type == "BCRC_uniform_only_zeros":
            uniform_rtg_mask = np.random.choice([0, 1], size=[num_seqs])
            for seq_idx, zero_out in enumerate(uniform_rtg_mask):
                if zero_out:
                    rtg_mask[seq_idx, :] = 0

        elif rtg_masking_type == "BCRC_uniform_first":
            uniform_rtg_mask = np.random.choice([0, 1], size=[num_seqs])
            rtg_mask[:, 0] = tt(uniform_rtg_mask)
            rtg_mask[:, 1:] = 0

        elif rtg_masking_type == "Unchanged":
            pass

        else:
            raise ValueError("rtg_masking_type not recognized")

        return rtg_mask

    def get_prediction_masks(self):
        """By default, predict everything that wan't present in the input"""
        # For last item prediction, the prediction masks will be exactly the opposite relative to the input masks
        s_mask = 1 - self.input_masks["*"]["state"]
        a_mask = 1 - self.input_masks["*"]["action"]
        rtg_mask = 1 - self.input_masks["*"]["rtg"]
        r_mask = 1 - self.input_masks["*"]["reward"]
        return {"*": {"state": s_mask.to(self.device), "action": a_mask.to(self.device), 
                        "rtg": rtg_mask.to(self.device), "reward": r_mask.to(self.device)}}

    @classmethod
    def must_have_size_multiple_of(cls, seq_len):
        """Batch should have a number of sequences multiple of the returned number"""
        return 1

    def num_maskings_per_type(self):
        """
        If the batch allows for more than one masking type, we want to be able to perfectly tile N maskings of each
        type in the batch in order to reduce variance. We use `must_have_size_multiple_of` to determine how many
        sequences we should mask with each masking type.
        """
        num_masking_types = self.must_have_size_multiple_of(self.seq_len)
        assert (
            self.num_seqs % num_masking_types == 0
        ), "Num seqs in the batch {} must be divisible by num_masking_types {}".format(self.num_seqs, num_masking_types)
        num_per_type = self.num_seqs // num_masking_types
        return num_per_type

    # @property
    # def model_input(self):
    #     inp, timestep_inp = self.input_data.model_input(self.input_masks)
    #     return inp, timestep_inp

    # def empty_input_masks(self):
    #     s_in_mask = torch.zeros((self.num_seqs, self.seq_len))
    #     act_in_mask = torch.zeros((self.num_seqs, self.seq_len))
    #     rtg_in_mask = torch.zeros((self.num_seqs, self.seq_len))
    #     return act_in_mask, rtg_in_mask, s_in_mask

    # def empty_pred_masks(self):
    #     s_mask = torch.zeros_like(self.input_masks["*"]["state"])
    #     a_mask = torch.zeros_like(self.input_masks["*"]["action"])
    #     r_mask = torch.zeros_like(self.input_masks["*"]["rtg"])
    #     return a_mask, r_mask, s_mask

    # ###############
    # # INPUT UTILS #
    # ###############

    # def get_factor(self, factor_name):
    #     return self.input_data.get_factor(factor_name)

    # def get_input_mask_for_factor(self, factor_name):
    #     if "*" in self.input_masks:
    #         mask_key = TokenSeq.get_mask_key(factor_name, self.input_masks["*"])
    #         return self.input_masks["*"][mask_key]
    #     else:
    #         raise NotImplementedError()

    # def get_masked_input_factor(self, factor_name, mask_nans=False):
    #     factor = self.get_factor(factor_name)
    #     input_mask = self.get_input_mask_for_factor(factor_name)
    #     return factor.mask(input_mask, mask_nans)

    # def get_prediction_mask_for_factor(self, factor_name):
    #     if "*" in self.prediction_masks:
    #         return self.prediction_masks["*"][factor_name]
    #     else:
    #         raise NotImplementedError()

    ###################

    # def add_model_output(self, batch_output):
    #     self.input_data.add_model_output(batch_output)
    #     self.computed_output = True

    # def compute_loss_and_acc(self, loss_weights):
    #     """
    #     We are given the output of the transformer
    #     We now want to computing losses and accuracy directly on a output head (predicting in behaviour space)

    #     NOTE: currently does not do accuracies
    #     """
    #     assert self.computed_output, "Have to add output with add_model_output before trying to compute loss"

    #     # need to implement a get_masked_items for Factors
    #     loss_dict = self.input_data.get_loss(loss_weights, self.prediction_masks)

    #     total_loss = 0.0
    #     for ts_name, factors_dict in loss_dict.items():
    #         for factor_name, v in factors_dict.items():
    #             total_loss += v.cpu()

    #     self.loss = total_loss
    #     loss_dict["total"] = total_loss
    #     return loss_dict

    @classmethod
    def get_dummy_batch_output(cls, data, batch_params, trainer):
        """
        Based on some data (in FullTokenSeq format), create a dummy batch and return it with the computed predictions

        TODO: have a parameter to do this with the model in eval mode, so as to not accidentally not use eval mode when
         evaluating
        """
        b = cls.from_params(data, batch_params)
        trainer.model(b)
        return b


class RandomPred(Batch):
    """
    Mask which has ======RND===== for actions
    and            ======RND===== for states
    """

    def __init__(self, mask_p=None, random_mask_p=True, **kwargs):
        """
        If using RandomPred, masks will always be random and you'll try to predict the randomly masked items.

        There will be a mask_p chance of having a mask at a position, and 1-mask_p chance of seeing the input.
        """
        super().__init__(**kwargs)
        # TODO: eventually re-introduce this?
        # assert self.rtg_masking_type in [
        #     "BCRC_uniform_all",
        #     "BCRC_uniform_first",
        # ], "If you're doing randomized masking, you probably want to train to do both BC and RC"

        self.random_mask_p = random_mask_p  # If random mask p, ignore mask probs
        if self.random_mask_p:
            assert mask_p is None, "If using random_mask_p, the mask_p should be set to None"
        else:
            assert 0 < mask_p < 1, "Mask p has to be a probability, and 0 or 1 don't make sense"
            self.mask_probs = [mask_p, 1 - mask_p]

        self.input_masks = self.get_input_masks()
        self.prediction_masks = self.get_prediction_masks()

    def get_input_masks(self):
        """
        Each token sequence will be of shape [num_seqs, seq_len, factor_size_sum]

        NOTE: Wherever the mask is _0_, the input will be masked out
        """
        mask_size = (self.num_seqs, self.seq_len)
        if self.random_mask_p:
            s_mask, a_mask, rtg_mask, r_mask = [], [], [], []
            for i in range(self.num_seqs): ## this is the batch size
                # NOTE: There are probably more efficient ways to do this.
                seq_mask_ps = np.random.uniform()
                seq_ps = [seq_mask_ps, 1 - seq_mask_ps]

                seq_s_mask = np.random.choice([0, 1], p=seq_ps, size=[self.seq_len])
                seq_a_mask = np.random.choice([0, 1], p=seq_ps, size=[self.seq_len])
                seq_rtg_mask = np.random.choice([0, 1], p=seq_ps, size=[self.seq_len])
                seq_r_mask = np.random.choice([0, 1], p=seq_ps, size=[self.seq_len])

                s_mask.append(seq_s_mask)
                a_mask.append(seq_a_mask)
                rtg_mask.append(seq_rtg_mask)
                r_mask.append(seq_r_mask)

            s_mask = np.array(s_mask)
            a_mask = np.array(a_mask)
            rtg_mask = np.array(rtg_mask)
            r_mask = np.array(r_mask)

        else:
            s_mask = np.random.choice([0, 1], p=self.mask_probs, size=mask_size)
            a_mask = np.random.choice([0, 1], p=self.mask_probs, size=mask_size)
            rtg_mask = np.random.choice([0, 1], p=self.mask_probs, size=mask_size)
            r_mask = np.random.choice([0, 1], p=self.mask_probs, size=mask_size)

        rtg_mask = self.postprocess_rtg_mask(rtg_mask, self.rtg_masking_type)
        return {"*": {"state": tt(s_mask).to(self.device), 
                        "action": tt(a_mask).to(self.device), 
                            "rtg": tt(rtg_mask).to(self.device),
                                "reward": tt(r_mask).to(self.device)}}

    def get_prediction_masks(self):
        """For item prediction, the prediction masks will be exactly the opposite relative to the input masks"""
        return super().get_prediction_masks()