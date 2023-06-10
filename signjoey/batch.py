# coding: utf-8
import math
import random
import torch
import numpy as np


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
        self,
        torch_batch,
        txt_pad_index,
        gla_pad_index=2,
        sgn_dim = 256,
        is_train: bool = False,
        use_cuda: bool = False,
        frame_subsampling_ratio: int = None,
        random_frame_subsampling: bool = None,
        random_frame_masking_ratio: float = None,
    ):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with sgn (sign),
        gls (gloss), and txt (text) length, masks, number of non-padded tokens in txt.
        Furthermore, it can be sorted by sgn length.

        :param torch_batch:
        :param txt_pad_index:
        :param sgn_dim:
        :param is_train:
        :param use_cuda:
        :param random_frame_subsampling
        """

        # Sequence Information
        self.sequence = torch_batch.sequence
        self.signer = torch_batch.signer
        # Sign
        self.sgn, self.sgn_lengths = torch_batch.sgn

        # Here be dragons
        if frame_subsampling_ratio:
            tmp_sgn = torch.zeros_like(self.sgn)
            tmp_sgn_lengths = torch.zeros_like(self.sgn_lengths)
            for idx, (features, length) in enumerate(zip(self.sgn, self.sgn_lengths)):
                features = features.clone()
                if random_frame_subsampling and is_train:
                    init_frame = random.randint(0, (frame_subsampling_ratio - 1))
                else:
                    init_frame = math.floor((frame_subsampling_ratio - 1) / 2)

                tmp_data = features[: length.long(), :]
                tmp_data = tmp_data[init_frame::frame_subsampling_ratio]
                tmp_sgn[idx, 0 : tmp_data.shape[0]] = tmp_data
                tmp_sgn_lengths[idx] = tmp_data.shape[0]

            self.sgn = tmp_sgn[:, : tmp_sgn_lengths.max().long(), :]
            self.sgn_lengths = tmp_sgn_lengths

        if random_frame_masking_ratio and is_train:
            tmp_sgn = torch.zeros_like(self.sgn)
            num_mask_frames = (
                (self.sgn_lengths * random_frame_masking_ratio).floor().long()
            )
            for idx, features in enumerate(self.sgn):
                features = features.clone()
                mask_frame_idx = np.random.permutation(
                    int(self.sgn_lengths[idx].long().numpy())
                )[: num_mask_frames[idx]]
                features[mask_frame_idx, :] = 1e-8
                tmp_sgn[idx] = features
            self.sgn = tmp_sgn

        self.sgn_dim = sgn_dim
        self.sgn_mask = (self.sgn != torch.zeros(sgn_dim))[..., 0].unsqueeze(1)

        # Text
        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        # Gloss
        self.gls = None
        self.gls_lengths = None
        # @jinhui
        self.gls_mask = None
        self.gls_input = None

        # Other
        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda
        self.num_seqs = self.sgn.size(0)

        if hasattr(torch_batch, "txt"):
            txt, txt_lengths = torch_batch.txt
            # txt_input is used for teacher forcing, last one is cut off
            self.txt_input = txt[:, :-1]
            self.txt_lengths = txt_lengths
            # txt is used for loss computation, shifted by one since BOS
            self.txt = txt[:, 1:]
            # we exclude the padded areas from the loss computation
            self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        if hasattr(torch_batch, "gls"):
            gls, gls_lengths = torch_batch.gls
            #self.num_gls_tokens = gls_lengths.sum().detach().clone().numpy()
            # @jinhui

            self.gls_input = gls
            self.gls_lengths = gls_lengths

            self.gls = gls
            # we exclude the padded areas from the loss computation
            self.gls_mask = (self.gls_input != gla_pad_index).unsqueeze(1)
            self.num_gls_tokens = (self.gls != gla_pad_index).data.sum().item()

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.sgn = self.sgn.cuda()
        self.sgn_mask = self.sgn_mask.cuda()

        if self.txt_input is not None:
            self.txt = self.txt.cuda()
            self.txt_mask = self.txt_mask.cuda()
            self.txt_input = self.txt_input.cuda()

        if self.gls_input is not None:
            self.gls = self.gls.cuda()
            self.gls_mask = self.gls_mask.cuda()
            self.gls_input = self.gls_input.cuda()

    def sort_by_sgn_lengths(self):
        """
        Sort by sgn length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        self.signer = [self.signer[pi] for pi in perm_index]
        self.sequence = [self.sequence[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]

            self.gls_mask = self.gls_mask[perm_index]
            self.gls_input = self.gls_input[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index


class Batch_Sign:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
            self,
            torch_batch,
            txt_pad_index,
            gla_pad_index=2,
            sgn_dim=256,
            is_train: bool = False,
            use_cuda: bool = False,
            frame_subsampling_ratio: int = None,
            random_frame_subsampling: bool = None,
            random_frame_masking_ratio: float = None,
            if_MixGen = None,
    ):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with sgn (sign),
        gls (gloss), and txt (text) length, masks, number of non-padded tokens in txt.
        Furthermore, it can be sorted by sgn length.

        :param torch_batch:
        :param txt_pad_index:
        :param sgn_dim:
        :param is_train:
        :param use_cuda:
        :param random_frame_subsampling
        """

        # Sequence Information
        self.sequence = torch_batch.sequence
        self.signer = torch_batch.signer
        # Sign
        self.sgn, self.sgn_lengths = torch_batch.sgn

        # Here be dragons
        if frame_subsampling_ratio:
            tmp_sgn = torch.zeros_like(self.sgn)
            tmp_sgn_lengths = torch.zeros_like(self.sgn_lengths)
            for idx, (features, length) in enumerate(zip(self.sgn, self.sgn_lengths)):
                features = features.clone()
                if random_frame_subsampling and is_train:
                    init_frame = random.randint(0, (frame_subsampling_ratio - 1))
                else:
                    init_frame = math.floor((frame_subsampling_ratio - 1) / 2)

                tmp_data = features[: length.long(), :]
                tmp_data = tmp_data[init_frame:frame_subsampling_ratio]
                tmp_sgn[idx, 0: tmp_data.shape[0]] = tmp_data
                tmp_sgn_lengths[idx] = tmp_data.shape[0]

            self.sgn = tmp_sgn[:, : tmp_sgn_lengths.max().long(), :]
            self.sgn_lengths = tmp_sgn_lengths

        if random_frame_masking_ratio and is_train:
            tmp_sgn = torch.zeros_like(self.sgn)
            num_mask_frames = (
                (self.sgn_lengths * random_frame_masking_ratio).floor().long()
            )
            for idx, features in enumerate(self.sgn):
                features = features.clone()
                mask_frame_idx = np.random.permutation(
                    int(self.sgn_lengths[idx].long().numpy())
                )[: num_mask_frames[idx]]
                features[mask_frame_idx, :] = 1e-8
                tmp_sgn[idx] = features
            self.sgn = tmp_sgn

        self.sgn_dim = sgn_dim
        self.sgn_mask = (self.sgn != torch.zeros(sgn_dim))[..., 0].unsqueeze(1)

        # Text
        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        # Gloss
        self.gls = None
        self.gls_lengths = None
        # @jinhui
        self.gls_mask = None
        self.gls_input = None

        # Other
        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda
        self.num_seqs = self.sgn.size(0)

        if hasattr(torch_batch, "txt"):
            txt, txt_lengths = torch_batch.txt
            # txt_input is used for teacher forcing, last one is cut off
            self.txt_input = txt[:, :-1]
            self.txt_lengths = txt_lengths
            # txt is used for loss computation, shifted by one since BOS
            self.txt = txt[:, 1:]
            # we exclude the padded areas from the loss computation
            self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        if hasattr(torch_batch, "gls"):
            gls, gls_lengths = torch_batch.gls
            # self.num_gls_tokens = gls_lengths.sum().detach().clone().numpy()
            # @jinhui
            ratio = int(max(torch.sum(self.sgn_lengths)// torch.sum(gls_lengths), 1))
            self.gls_input = gls.repeat_interleave(ratio, dim=1)
            self.gls_input_lengths = gls_lengths * ratio
            self.gls_input_mask = (self.gls_input != gla_pad_index).unsqueeze(1)

            # self.gls_input = gls
            #
            # self.gls_input_mask = (self.gls_input != gla_pad_index).unsqueeze(1)
            # self.gls_input_lengths = gls_lengths
            #
            self.gls = gls
            self.gls_lengths = gls_lengths
            # we exclude the padded areas from the loss computation
            self.gls_mask = (self.gls != gla_pad_index).unsqueeze(1)
            self.num_gls_tokens = (self.gls != gla_pad_index).data.sum().item()


        # TODO concat samples
        if if_MixGen == 1:

            random_indexs_1 = torch.tensor(list(range(0, torch_batch.batch_size)))
            random_indexs_2 = torch.tensor(list(range(0, torch_batch.batch_size)))
            random.shuffle(random_indexs_1)
            random.shuffle(random_indexs_2)
            # TODO cat samples

            # sign
            self.sgn = self.cat_sequence(input=self.sgn, index_1=random_indexs_1,index_2=random_indexs_2)
            self.sgn_mask = self.cat_sequence(input=self.sgn_mask, dim=2, index_1=random_indexs_1,index_2=random_indexs_2)
            self.sgn_lengths = self.cat_lengths(input=self.sgn_lengths, index_1=random_indexs_1,index_2=random_indexs_2)

            for index in range(0,torch_batch.batch_size):
                _signer_a = self.signer[random_indexs_1[index]]
                _signer_b = self.signer[random_indexs_2[index]]
                self.signer.append(_signer_a + f"${index}$" + _signer_b)

                _sequence_a = self.sequence[random_indexs_1[index]]
                _sequence_b = self.sequence[random_indexs_2[index]]
                self.sequence.append(_sequence_a + f"${index}$" + _sequence_b)

            if self.gls is not None:
                self.gls = self.cat_sequence(input=self.gls, index_1=random_indexs_1,index_2=random_indexs_2)
                self.gls_lengths = self.cat_lengths(input=self.gls_lengths, index_1=random_indexs_1,index_2=random_indexs_2)

                self.gls_input_mask = self.cat_sequence(input=self.gls_input_mask, dim=2, index_1=random_indexs_1,index_2=random_indexs_2)
                self.gls_input = self.cat_sequence(input=self.gls_input, index_1=random_indexs_1,index_2=random_indexs_2)
                self.gls_input_lengths = self.cat_lengths(input=self.gls_input_lengths, index_1=random_indexs_1,index_2=random_indexs_2)

            if self.txt is not None:
                self.txt =  self.cat_sequence(input=self.txt, index_1=random_indexs_1,index_2=random_indexs_2)
                self.txt_mask =  self.cat_sequence(input=self.txt_mask, dim=2, index_1=random_indexs_1,index_2=random_indexs_2)
                self.txt_input =  self.cat_sequence(input=self.txt_input, index_1=random_indexs_1,index_2=random_indexs_2)
                self.txt_lengths =  self.cat_lengths(input=self.txt_lengths, index_1=random_indexs_1,index_2=random_indexs_2)


        if use_cuda:
            self._make_cuda()

    def cat_sequence(self, input, index_1, index_2, dim=1):

        temp_A = input[index_1]
        temp_B = input[index_2]
        temp_cat = torch.cat((temp_A,temp_B), dim=dim)

        return temp_cat

    def cat_lengths(self, input, index_1, index_2):
        temp_A = input[index_1]
        temp_B = input[index_2]
        temp_cat = temp_A + temp_B

        return temp_cat

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.sgn = self.sgn.cuda()
        self.sgn_mask = self.sgn_mask.cuda()

        if self.txt_input is not None:
            self.txt = self.txt.cuda()
            self.txt_mask = self.txt_mask.cuda()
            self.txt_input = self.txt_input.cuda()

        if self.gls_input is not None:
            self.gls = self.gls.cuda()
            self.gls_mask = self.gls_mask.cuda()
            self.gls_input = self.gls_input.cuda()
            self.gls_input_mask = self.gls_input_mask.cuda()

    def sort_by_sgn_lengths(self):
        """
        Sort by sgn length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        # self.signer = [self.signer[pi] for pi in perm_index] # not important
        self.sequence = [self.sequence[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]
            self.gls_mask = self.gls_mask[perm_index]

            self.gls_input_mask = self.gls_input_mask[perm_index]
            self.gls_input = self.gls_input[perm_index]
            self.gls_input_lengths = self.gls_input_lengths[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        return rev_index

class Batch_gls2text:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
            self,
            torch_batch,
            txt_pad_index,
            gla_pad_index=2,
            is_train: bool = False,
            use_cuda: bool = False,
            frame_subsampling_ratio: int = None,
            random_frame_subsampling: bool = None,
            random_frame_masking_ratio: float = None,
            if_MixGen = 0,
    ):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with sgn (sign),
        gls (gloss), and txt (text) length, masks, number of non-padded tokens in txt.
        Furthermore, it can be sorted by sgn length.

        :param torch_batch:
        :param txt_pad_index:
        :param sgn_dim:
        :param is_train:
        :param use_cuda:
        :param random_frame_subsampling
        """


        # Text
        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        # Gloss
        self.gls = None
        self.gls_lengths = None
        # @jinhui
        self.gls_mask = None
        self.gls_input = None

        # Other
        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda

        if hasattr(torch_batch, "txt"):
            txt, txt_lengths = torch_batch.txt
            # txt_input is used for teacher forcing, last one is cut off
            self.txt_input = txt[:, :-1]
            self.txt_lengths = txt_lengths
            # txt is used for loss computation, shifted by one since BOS
            self.txt = txt[:, 1:]
            # we exclude the padded areas from the loss computation
            self.txt_mask = (self.txt_input != txt_pad_index).unsqueeze(1)
            self.num_txt_tokens = (self.txt != txt_pad_index).data.sum().item()

        # if hasattr(torch_batch, "gls"):
        #     gls, gls_lengths = torch_batch.gls
        #     # self.num_gls_tokens = gls_lengths.sum().detach().clone().numpy()
        #     # @jinhui
        #
        #     self.gls_input = gls
        #     self.gls_lengths = gls_lengths
        #
        #     self.gls = gls
        #     # we exclude the padded areas from the loss computation
        #     self.gls_mask = (self.gls_input != gla_pad_index).unsqueeze(1)
        #     self.num_gls_tokens = (self.gls != gla_pad_index).data.sum().item()
        #
        # self.num_seqs = self.gls.size(0)
        # TODO @jinhui 使用——
        if hasattr(torch_batch, "gls"):
            gls, gls_lengths = torch_batch.gls
            # self.num_gls_tokens = gls_lengths.sum().detach().clone().numpy()
            # @jinhui
            ratio = random.randint(4,8)
            self.gls_input = gls.repeat_interleave(ratio, dim=1)
            self.gls_input_lengths = gls_lengths * ratio
            self.gls_input_mask = (self.gls_input != gla_pad_index).unsqueeze(1)

            self.gls = gls
            # we exclude the padded areas from the loss computation
            self.gls_mask = (self.gls != gla_pad_index).unsqueeze(1)
            self.num_gls_tokens = (self.gls != gla_pad_index).data.sum().item()
            self.gls_lengths = gls_lengths
        self.num_seqs = self.gls.size(0)
        # #TODO extend gloss to sign
        # import torch
        #
        # gls, gls_lengths = torch_batch.gls
        #
        # # 用 repeat_interleave 实现在序列长度维度上重复6次
        # self.gls_input_extend = gls.repeat_interleave(6, dim=1)
        #
        # # 更新 gls_lengths_extend 以适应复制后的 gls_input_extend
        # self.gls_lengths_extend = gls_lengths * 6
        #
        # self.gls_extend = self.gls_input_extend
        #
        # # 更新 gls_mask_extend 的计算以适应复制后的 gls_input_extend
        # self.gls_mask_extend = (self.gls_input_extend != gla_pad_index).unsqueeze(1)
        #
        # # 使用复制后的 gls_extend 计算 num_gls_tokens_extend
        # self.num_gls_tokens_extend = (self.gls_extend != gla_pad_index).data.sum().item()

        # TODO concat samples
        if if_MixGen == 1:

            random_indexs_1 = torch.tensor(list(range(0, torch_batch.batch_size)))
            random_indexs_2 = torch.tensor(list(range(0, torch_batch.batch_size)))
            random.shuffle(random_indexs_1)
            random.shuffle(random_indexs_2)
            # TODO cat samples

            if self.gls is not None:
                self.gls = self.cat_sequence(input=self.gls, index_1=random_indexs_1,index_2=random_indexs_2)
                self.gls_lengths = self.cat_lengths(input=self.gls_lengths, index_1=random_indexs_1,index_2=random_indexs_2)

                self.gls_mask = self.cat_sequence(input=self.gls_mask, dim=2, index_1=random_indexs_1,index_2=random_indexs_2)
                self.gls_input = self.cat_sequence(input=self.gls_input, index_1=random_indexs_1,index_2=random_indexs_2)

            if self.txt is not None:
                self.txt =  self.cat_sequence(input=self.txt, index_1=random_indexs_1,index_2=random_indexs_2)
                self.txt_mask =  self.cat_sequence(input=self.txt_mask, dim=2, index_1=random_indexs_1,index_2=random_indexs_2)
                self.txt_input =  self.cat_sequence(input=self.txt_input, index_1=random_indexs_1,index_2=random_indexs_2)
                self.txt_lengths =  self.cat_lengths(input=self.txt_lengths, index_1=random_indexs_1,index_2=random_indexs_2)


        if use_cuda:
            self._make_cuda()

    def cat_sequence(self, input, index_1, index_2, dim=1):

        temp_A = input[index_1]
        temp_B = input[index_2]
        temp_cat = torch.cat((temp_A,temp_B), dim=dim)

        return temp_cat

    def cat_lengths(self, input, index_1, index_2):
        temp_A = input[index_1]
        temp_B = input[index_2]
        temp_cat = temp_A + temp_B

        return temp_cat

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """

        if self.txt_input is not None:
            self.txt = self.txt.cuda()
            self.txt_mask = self.txt_mask.cuda()
            self.txt_input = self.txt_input.cuda()

        if self.gls_input is not None:
            self.gls = self.gls.cuda()
            self.gls_mask = self.gls_mask.cuda()
            self.gls_input = self.gls_input.cuda()
            self.gls_input_mask = self.gls_input_mask.cuda()

    def sort_by_sgn_lengths(self):
        """
        Sort by sgn length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        self.signer = [self.signer[pi] for pi in perm_index]
        self.sequence = [self.sequence[pi] for pi in perm_index]

        if self.gls is not None:
            self.gls = self.gls[perm_index]
            self.gls_lengths = self.gls_lengths[perm_index]
            self.gls_mask = self.gls_mask[perm_index]

            self.gls_input_mask = self.gls_input_mask[perm_index]
            self.gls_input = self.gls_input[perm_index]
            self.gls_input_lengths = self.gls_input_lengths[perm_index]

        if self.txt is not None:
            self.txt = self.txt[perm_index]
            self.txt_mask = self.txt_mask[perm_index]
            self.txt_input = self.txt_input[perm_index]
            self.txt_lengths = self.txt_lengths[perm_index]

        if self.use_cuda:
            self._make_cuda()

        return rev_index