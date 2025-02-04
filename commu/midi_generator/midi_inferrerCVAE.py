import math
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import yacs.config

from logger import logger
from commu.midi_generator.container import TransXlInputData
from commu.model.Transformer_CVAE import Transformer_CVAE
from commu.preprocessor.encoder import TOKEN_OFFSET
from commu.preprocessor.utils.constants import DEFAULT_POSITION_RESOLUTION


class TeacherForceTask:
    def __init__(self, input_data):
        self.input_data = input_data
        self.next_tokens_forced = []
        self.wrong_tokens = []
        self.no_sequence_appended = False
        self.is_incomplete = input_data.num_measures % 4 != 0
        self.incomplete_filled = not self.is_incomplete

        self.chord_token, self.chord_position = input_data.chord_token_components.values()
        assert len(self.chord_token) == len(self.chord_position), "Wrong Chord Length"
        self.chord_length = len(self.chord_token)
        self.inter_chord_flags = []
        for i in self.chord_position:
            if i == TOKEN_OFFSET.POSITION.value:
                self.inter_chord_flags.append(False)
            else:
                self.inter_chord_flags.append(True)

    def check_first_position(self, seq):
        """
        check if it's a token following a bar token
        """
        return self.incomplete_filled and seq[-1] == TOKEN_OFFSET.BAR.value

    def check_remnant_chord(self):
        """
        check if there any more chords to write
        if not, return False
        """
        return bool(len(self.chord_token) * len(self.chord_position))

    def check_length_fit(self):
        """
        check if one chord per bar needed
        """
        return self.chord_length == int(self.input_data.num_measures // 4 * 4)

    def check_position_fit(self, seq):
        """
        check if a chord token needs to be filled next
        """
        return seq[-2] == TOKEN_OFFSET.BAR.value and seq[-1] == TOKEN_OFFSET.POSITION.value

    def check_one_chord_per_bar_case(self, seq):
        """
        case: one chord per bar
        """
        return (
            self.check_remnant_chord()
            and self.incomplete_filled
            and self.check_length_fit()
            and self.check_position_fit(seq)
        )

    def check_mul_chord_per_bar_case(self, seq):
        """
        case: multiple chords per bar
        """
        is_first_position_chord = (
            self.check_remnant_chord()
            and self.incomplete_filled
            and not self.check_length_fit()
            and self.check_position_fit(seq)
        )

        is_inter_position_chord = (
            self.check_remnant_chord()
            and self.incomplete_filled
            and not self.check_length_fit()
            and not self.check_position_fit(seq)
            and seq[-1] == self.chord_position[0]
            and self.inter_chord_flags[0]
        )
        return is_first_position_chord or is_inter_position_chord

    def check_chord_position_passed(self, token):
        """
        in case a generated token skipped necessary position
        """
        if not self.check_remnant_chord():
            return False
        is_position_passed = (
            self.chord_position[0] < token < TOKEN_OFFSET.POSITION.value + DEFAULT_POSITION_RESOLUTION
            or token == TOKEN_OFFSET.BAR.value
        )
        return self.inter_chord_flags[0] and is_position_passed

    def check_wrong_chord_token_generated(self, token):
        """
        all chord tokens should be teacher forced
        """
        return TOKEN_OFFSET.CHORD_START.value <= token <= TOKEN_OFFSET.CHORD_END.value

    def check_wrong_eos_generated(self, token):
        return self.check_remnant_chord() and token == TOKEN_OFFSET.EOS.value

    def check_wrong_bar_token_generated(self, token):
        return not self.check_remnant_chord() and token == TOKEN_OFFSET.BAR.value

    def teach_first_position(self) -> None:
        """
        teach 1/128 position right after a bar token
        """
        self.next_tokens_forced.append(int(TOKEN_OFFSET.POSITION.value))

    def teach_chord_token(self):
        next_chord_tokens = self.chord_token.pop(0)
        print(f"Teach token is: {next_chord_tokens}, num of left tokens: {len(self.chord_token)}")
        self.next_tokens_forced.append(next_chord_tokens)
        self.chord_position.pop(0)
        self.inter_chord_flags.pop(0)
        self.wrong_tokens = []

    def teach_chord_position(self):
        next_position_token = self.chord_position[0]
        self.next_tokens_forced.append(next_position_token)
        self.wrong_tokens = []

    def teach_wrong_chord_token(self, wrong_token):
        self.no_sequence_appended = True
        self.wrong_tokens.append(wrong_token)

    def teach_remnant_chord(self):
        token = self.chord_position[0] if self.inter_chord_flags[0] else TOKEN_OFFSET.BAR.value
        self.next_tokens_forced.append(token)

    def teach_eos(self):
        token = TOKEN_OFFSET.EOS.value
        self.next_tokens_forced.append(token)

    def validate_teacher_forced_sequence(self, seq) -> None:
        def _count_num_chord(seq):
            chord_counter = 0
            for token in seq:
                if TOKEN_OFFSET.CHORD_START.value <= token <= TOKEN_OFFSET.CHORD_END.value:
                    chord_counter += 1
            return chord_counter

        num_bars = seq.count(TOKEN_OFFSET.BAR.value)
        num_chord = _count_num_chord(seq)

        if len(self.chord_token) != 0:
            raise Exception(
                f"remnant chord length: {len(self.chord_token)} \n" "error in teacher forcing"
            )
        elif num_bars != int(math.ceil(self.input_data.num_measures)):
            raise Exception(f"bar length: {num_bars} \n" "error in bar length")
        elif num_chord != self.chord_length:
            raise Exception(
                f"num_chord: {num_chord} vs {self.chord_length} \n" "error in chord length"
            )
        else:
            logger.info(f"correct_length: {num_bars}")
            logger.info(seq)


class InferenceTask:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(
        self,
        model: Transformer_CVAE,
        input_data: TransXlInputData,
        inference_cfg: yacs.config.CfgNode,
    ):
        self.model = model
        self.input_data = input_data
        self.inference_cfg = inference_cfg
    
    def generate_latent_init_seq(self):
        latent_dim = self.inference_cfg.GENERATION.latent_dim
        seq_length = self.inference_cfg.GENERATION.seq_length
        latent = torch.normal(0, 1, (seq_length + 11, 1, latent_dim)).to(self.device)
        seq = (torch.ones(seq_length, 1).long() * self.inference_cfg.GENERATION.pad_index).to(self.device)
        return latent, seq

    def update_seq_tensor(self, seq, seq_tensor):
        seq_length = self.inference_cfg.GENERATION.seq_length
        cur_pos = len(seq) - 1
        if cur_pos >= seq_length:
            tmp = seq_tensor[1:, 0].clone()
            seq_tensor[:-1, 0] = tmp
            cur_pos = -1
        seq_tensor[cur_pos, 0] = seq[-1]
        return seq_tensor, cur_pos

    def calc_logits(
        self, seq, seq_tensor: torch.Tensor, latent: torch.Tensor, meta: torch.Tensor
    ) -> torch.Tensor:
        seq_tensor, cur_pos = self.update_seq_tensor(seq, seq_tensor)
        seq_tensor = torch.concat((meta, seq_tensor), dim=0)
        logits = self.model.forward_generate(seq_tensor, latent)[cur_pos, 0]
        return logits

    def calc_probs(self, logits):
        # Handle temp 0 (argmax) case
        if self.input_data.temperature == 0:
            probs = torch.zeros_like(logits)
            probs[logits.argmax()] = 1.0
        else:
            # Apply temperature spec
            logits /= self.input_data.temperature
            # Compute softmax
            # Use Gumbel Softmax Trick
            probs = F.softmax(logits / self.inference_cfg.GENERATION.tau, dim=-1)

        probs = F.pad(probs, [1, 0])
        return probs

    def apply_sampling(self, probs, wrong_tokens):
        _, top_idx = torch.topk(probs, self.input_data.top_k)
        mask = torch.zeros_like(probs)
        mask[top_idx] = 1.0
        if wrong_tokens:
            for w in wrong_tokens:
                mask[w] = 0.0
        probs *= mask
        probs /= torch.sum(probs)
        return probs

    def infer_token(self, probs):
        token = torch.multinomial(probs, 1)
        token = int(token.item())
        return token

    def generate_sequence(self, seq_tensor, latent, meta):
        logits = None
        teacher = TeacherForceTask(self.input_data)
        first_loop = True
        seq = [self.inference_cfg.GENERATION.pad_index, TOKEN_OFFSET.BAR.value]
        for _ in range(self.inference_cfg.GENERATION.generation_length):
            if seq[-1] == 1:
                break

            if teacher.next_tokens_forced:
                seq_tensor, _ = self.update_seq_tensor(seq, seq_tensor)
                next_token = teacher.next_tokens_forced.pop(0)
                seq.append(next_token)
                logits = self.calc_logits(seq, seq_tensor, latent, meta)
                continue

            if teacher.no_sequence_appended:
                assert logits is not None
                teacher.no_sequence_appended = False
            elif first_loop:
                logits = self.calc_logits(seq, seq_tensor, latent, meta)
                first_loop = False
            else:
                logits = self.calc_logits(seq, seq_tensor, latent, meta)

            probs = self.calc_probs(logits)
            probs = self.apply_sampling(probs, teacher.wrong_tokens)

            # teacher forcing
            # in case with incomplete measure, trigger a flag after second bar token
            if not teacher.incomplete_filled:
                teacher.incomplete_filled = True if seq.count(TOKEN_OFFSET.BAR.value) > 1 else False

            # forcefully assign position 1/128 right after bar token
            if teacher.check_first_position(seq):
                teacher.teach_first_position()
                continue

            # in case there is one chord per bar
            if teacher.check_one_chord_per_bar_case(seq):
                teacher.teach_chord_token()
                continue

            # in case the chord changes within a bar
            if teacher.check_mul_chord_per_bar_case(seq):
                teacher.teach_chord_token()
                continue

            # teacher forcing followed by token inference so that we can check if the wrong token was generated
            try:
                token = self.infer_token(probs)
                print(token)
            except RuntimeError as e:
                logger.error(f"Sampling Error: {e}")
                seq = None
                break

            # generated token skipped necessary position
            if teacher.check_chord_position_passed(token):
                teacher.teach_chord_position()
                continue

            # wrong chord token generated
            if teacher.check_wrong_chord_token_generated(token):
                teacher.teach_wrong_chord_token(token)
                continue

            # eos generated but we got more chords to write
            if teacher.check_wrong_eos_generated(token):
                teacher.teach_remnant_chord()
                continue

            # bar token generated but num measures exceed
            if teacher.check_wrong_bar_token_generated(token):
                teacher.teach_eos()
                continue

            seq.append(token)

        try:
            teacher.validate_teacher_forced_sequence(seq)
        except Exception as error_message:
            logger.error(error_message)
            seq = None
        return seq

    def validate_generated_sequence(self, seq: List[int]) -> bool:
        num_note = 0
        for idx, token in enumerate(seq):
            if idx + 2 > len(seq) - 1:
                break
            if token in range(TOKEN_OFFSET.NOTE_VELOCITY.value, TOKEN_OFFSET.CHORD_START.value):
                if (
                    seq[idx - 1] in range(TOKEN_OFFSET.POSITION.value, TOKEN_OFFSET.BPM.value)
                    and seq[idx + 1]
                    in range(TOKEN_OFFSET.PITCH.value, TOKEN_OFFSET.NOTE_VELOCITY.value)
                    and seq[idx + 2]
                    in range(TOKEN_OFFSET.NOTE_DURATION.value, TOKEN_OFFSET.POSITION.value)
                ):
                    num_note += 1
        return num_note > 0

    def execute(self, encoded_meta) -> List[List[int]]:
        idx = 0
        sequences = []
        while idx != self.input_data.num_generate:
            with torch.no_grad():
                logger.info("Generating the idx: " + str(idx + 1))
                meta = torch.LongTensor(encoded_meta).view(-1, 1).to(self.device)
                latent, seq_tensor = self.generate_latent_init_seq()
                seq = [self.inference_cfg.GENERATION.pad_index] # 0 is pad_idx
                seq = self.generate_sequence(seq_tensor, latent, meta)
                seq = [0] + encoded_meta + seq[1:]
                if seq is None:
                    continue
                if not self.validate_generated_sequence(seq):
                    logger.error("Empty sequence generated")
                    continue
            sequences.append(seq)
            idx += 1
        return sequences
