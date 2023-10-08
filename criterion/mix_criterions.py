import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torchmetrics


def item(tensor):
    if torch.is_tensor(tensor) and tensor.device.type == "xla":
        return tensor.detach()
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


class GuidedCrossEntMultiTaskCriterion(_Loss):
    def __init__(
            self,
            args,
            padding_idx,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.spch_loss_weight = args.spch_loss_weight
        self.asr_loss_weight = args.asr_loss_weight
        self.text_loss_weight = args.text_loss_weight
        self.use_cml = args.use_cml
        self.use_erm = args.use_erm
        self.cml_loss_weight = args.cml_loss_weight
        self.erm_loss_weight = args.erm_loss_weight
        self.alpha = args.guide_alpha
        self.text_alpha = args.text_alpha
        assert 0 <= self.alpha <= 1.0

        self.metrics = nn.ModuleDict({})
        if self.spch_loss_weight > 0.0:
            self.metrics.update({
                'speech_nll_loss': torchmetrics.SumMetric(compute_on_step=False),
                'speech_correct': torchmetrics.SumMetric(compute_on_step=False),
                'speech_total': torchmetrics.SumMetric(compute_on_step=False),
            })
        if self.asr_loss_weight > 0.0:
            self.metrics.update({
                'asr_nll_loss': torchmetrics.SumMetric(compute_on_step=False),
                'asr_correct': torchmetrics.SumMetric(compute_on_step=False),
                'asr_total': torchmetrics.SumMetric(compute_on_step=False),
            })
        if self.text_loss_weight > 0.0:
            self.metrics.update({
                'text_nll_loss': torchmetrics.SumMetric(compute_on_step=False),
                'text_correct': torchmetrics.SumMetric(compute_on_step=False),
                'text_total': torchmetrics.SumMetric(compute_on_step=False),
            })
        if self.alpha > 0.0:
            self.metrics.update({
                'guide_loss': torchmetrics.SumMetric(compute_on_step=False),
            })

        if args.use_cml:
            self.metrics.update({
                'mix_speech_nll_loss': torchmetrics.SumMetric(compute_on_step=False),
                'mix_asr_nll_loss': torchmetrics.SumMetric(compute_on_step=False),
                'mix_mlm_nll_loss': torchmetrics.SumMetric(compute_on_step=False),
                'mix_speech_correct': torchmetrics.SumMetric(compute_on_step=False),
                'mix_asr_correct': torchmetrics.SumMetric(compute_on_step=False),
                'mix_mlm_correct': torchmetrics.SumMetric(compute_on_step=False),
                'mix_speech_total': torchmetrics.SumMetric(compute_on_step=False),
                'mix_asr_total': torchmetrics.SumMetric(compute_on_step=False),
                'mix_mlm_total': torchmetrics.SumMetric(compute_on_step=False),
            })
        if self.use_erm:
            self.metrics.update({
                'reg_loss': torchmetrics.SumMetric(compute_on_step=False),
                'reg_total': torchmetrics.SumMetric(compute_on_step=False),
            })

    def forward(self, model, sample, reduce=True):
        reduction = 'sum' if reduce else 'none'
        net_input = sample["net_input"]
        output = model(**net_input)
        decoder_output = output[0]

        targets = {'transcript': sample["target"][0], 'translate': sample["target"][1]}

        logits = decoder_output[0]

        speech_loss, speech_nll_loss, speech_correct, speech_total = {}, {}, {}, {}
        text_loss, text_nll_loss, text_correct, text_total = {}, {}, {}, {}
        nll_loss, acc = {}, {}

        # asr
        spch_logits, _ = logits['transcript']
        if self.asr_loss_weight > 0.0:
            speech_loss['transcript'], speech_nll_loss['transcript'], speech_correct['transcript'], speech_total[
                'transcript'], _ = self.compute_loss_and_acc(model, spch_logits, targets['transcript'],
                                                             reduction=reduction)
            nll_loss['asr'] = speech_nll_loss['transcript'] / speech_total['transcript']
            acc['asr'] = speech_correct['transcript'] / speech_total['transcript'] * 100.0

        spch_logits, text_logits = logits['translate']
        # mt
        if self.text_loss_weight > 0.0:
            if 'reg_logits' in decoder_output[1] and self.text_alpha > 0.0:
                text_loss['translate'], text_nll_loss['translate'], text_correct['translate'], text_total[
                    'translate'], _ = self.guide_loss_and_acc(model, text_logits, decoder_output[1]['reg_logits'],
                                                              targets['translate'], reduction=reduction,
                                                              alpha=self.text_alpha)
            else:
                text_loss['translate'], text_nll_loss['translate'], text_correct['translate'], text_total[
                    'translate'], _ = self.compute_loss_and_acc(model, text_logits, targets['translate'],
                                                                reduction=reduction)
            nll_loss['text_translate'] = text_nll_loss['translate'] / text_total['translate']
            acc['text_translate'] = text_correct['translate'] / text_total['translate'] * 100.0
        # st
        if self.spch_loss_weight > 0.0:
            if text_logits is not None:
                speech_loss['translate'], speech_nll_loss['translate'], speech_correct['translate'], speech_total[
                    'translate'], guide_loss = self.guide_loss_and_acc(model, spch_logits, text_logits,
                                                                       targets['translate'], reduction=reduction)
            else:
                speech_loss['translate'], speech_nll_loss['translate'], speech_correct['translate'], speech_total[
                    'translate'], _ = self.compute_loss_and_acc(model, spch_logits, targets['translate'],
                                                                reduction=reduction)
            nll_loss['speech_translate'] = speech_nll_loss['translate'] / speech_total['translate']
            acc['speech_translate'] = speech_correct['translate'] / speech_total['translate'] * 100.0

        if not self.training:
            if self.asr_loss_weight > 0.0:
                self.metrics['asr_nll_loss'].update(speech_loss['transcript'])
                self.metrics['asr_correct'].update(speech_correct['transcript'])
                self.metrics['asr_total'].update(speech_total['transcript'])
            if self.text_loss_weight > 0.0:
                self.metrics['text_nll_loss'].update(text_loss['translate'])
                self.metrics['text_correct'].update(text_correct['translate'])
                self.metrics['text_total'].update(text_total['translate'])
            if self.spch_loss_weight > 0.0:
                self.metrics['speech_nll_loss'].update(speech_loss['translate'])
                self.metrics['speech_correct'].update(speech_correct['translate'])
                self.metrics['speech_total'].update(speech_total['translate'])
            if self.alpha > 0.0:
                self.metrics['guide_loss'].update(guide_loss)

        total_loss = 0.0
        if self.asr_loss_weight > 0.0:
            total_loss += speech_loss['transcript'] * self.asr_loss_weight
        if self.text_loss_weight > 0.0:
            total_loss += text_loss['translate'] * self.text_loss_weight
        if self.spch_loss_weight > 0.0:
            total_loss += speech_loss['translate'] * self.spch_loss_weight

        translate_logits_teacher = logits['translate'][1]

        mix_logits = decoder_output[1]['mix_dec_outs']
        erm_loss = decoder_output[1]['erm_loss']

        if self.use_cml and mix_logits is not None:
            mlm_mask = decoder_output[1]['mlm_mask']
            for task in ['transcript', 'translate']:
                if len(mix_logits[task]) == 0:
                    continue
                spch_logits, text_logits = mix_logits[task]
                if task == 'translate':
                    speech_loss[task], speech_nll_loss[task], speech_correct[task], speech_total[
                        task], _ = self.guide_loss_and_acc(model, spch_logits, translate_logits_teacher, targets[task],
                                                           reduction=reduction)
                else:
                    speech_loss[task], speech_nll_loss[task], speech_correct[task], speech_total[
                        task], _ = self.compute_loss_and_acc(model, spch_logits, targets[task], reduction=reduction)
                    mlm_target = targets[task].detach().clone()
                    mlm_target[~mlm_mask] = self.padding_idx
                    text_loss[task], text_nll_loss[task], text_correct[task], text_total[
                        task], _ = self.compute_loss_and_acc(model, text_logits, mlm_target, reduction=reduction)

            nll_loss['mix_speech_translate'] = speech_nll_loss['translate'] / speech_total['translate']
            nll_loss['mix_asr'] = speech_nll_loss['transcript'] / speech_total['transcript']
            nll_loss['mix_mlm'] = text_nll_loss['transcript'] / text_total['transcript']
            acc['mix_speech_translate'] = speech_correct['translate'] / speech_total['translate'] * 100.0
            acc['mix_asr'] = speech_correct['transcript'] / speech_total['transcript'] * 100.0
            acc['mix_mlm'] = text_correct['transcript'] / text_total['transcript'] * 100.0

            if not self.training:
                self.metrics['mix_speech_nll_loss'].update(speech_nll_loss['translate'])
                self.metrics['mix_asr_nll_loss'].update(speech_nll_loss['transcript'])
                self.metrics['mix_mlm_nll_loss'].update(text_nll_loss['transcript'])
                self.metrics['mix_speech_correct'].update(speech_correct['translate'])
                self.metrics['mix_asr_correct'].update(speech_correct['transcript'])
                self.metrics['mix_mlm_correct'].update(text_correct['transcript'])
                self.metrics['mix_speech_total'].update(speech_total['translate'])
                self.metrics['mix_asr_total'].update(speech_total['transcript'])
                self.metrics['mix_mlm_total'].update(text_total['transcript'])

            cml_loss = (speech_loss['transcript'] + speech_loss['translate'] + text_loss['transcript']) / 3

            if self.use_erm and erm_loss is not None:
                src_token_num = item(erm_loss.ne(0).sum())
                erm_loss = erm_loss.sum()
                cml_loss += erm_loss * self.erm_loss_weight
                if not self.training:
                    self.metrics['reg_loss'].update(erm_loss)
                    self.metrics['reg_total'].update(src_token_num)
                erm_loss /= src_token_num
            else:
                erm_loss = 0

            total_loss += cml_loss * self.cml_loss_weight

        mean_guide_loss = guide_loss / speech_total['translate'] if self.alpha > 0.0 else None
        logging_output = self.step_logging_output(
            acc, nll_loss, mean_guide_loss, erm_loss
        )

        return total_loss, logging_output, output

    def reduce_metric(self):
        output = {}
        for k, v in self.metrics.items():
            if 'total' in k:
                continue
            elif 'nll_loss' in k:
                output[k] = v.compute() / self.metrics[k.replace('nll_loss', 'total')].compute()
            elif 'correct' in k:
                output[k.replace('correct', 'acc')] = v.compute() / self.metrics[
                    k.replace('correct', 'total')].compute() * 100.0
        if self.alpha > 0.0:
            output['guide_loss'] = self.metrics['guide_loss'].compute() / self.metrics['speech_total'].compute()
        if self.use_cml and self.use_erm:
            output['reg_loss'] = self.metrics['reg_loss'].compute() / self.metrics['reg_total'].compute()
        for k, v in self.metrics.items():
            v.reset()
        return output

    def compute_loss_and_acc(self, model, logits, target, reduction='sum'):
        logits = logits.view(-1, logits.size(-1)).float()  # -> (B x T) x C
        target = target.view(-1)
        loss = F.cross_entropy(
            logits, target, ignore_index=self.padding_idx, reduction=reduction,
        )

        nll_loss = F.cross_entropy(
            logits, target, label_smoothing=0, ignore_index=self.padding_idx, reduction=reduction,
        ).detach()

        mask = target.ne(self.padding_idx)
        correct = torch.sum(logits.argmax(1).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)
        return loss, nll_loss, correct, total, torch.tensor(0.0)

    def guide_loss_and_acc(self, model, logits, logits_teacher, target, reduction='sum', alpha=None):
        """ lprobs_teacher is used as guide for lprobs """
        alpha = self.alpha if alpha is None else alpha
        if alpha == 0.0:
            return self.compute_loss_and_acc(model, logits, target, reduction=reduction)

        logits = logits.view(-1, logits.size(-1)).float()  # -> (B x T) x C
        logits_teacher = logits_teacher.view(-1, logits_teacher.size(-1)).float()  # -> (B x T) x C
        target = target.view(-1)
        loss = F.cross_entropy(logits, target, ignore_index=self.padding_idx, reduction=reduction)
        nll_loss = loss
        probs_teacher = F.softmax(logits_teacher, dim=-1).masked_fill_(target.unsqueeze(-1).eq(self.padding_idx), 0)
        probs_teacher = probs_teacher.detach()
        lprobs = F.log_softmax(logits, dim=-1)
        guide_loss = -(probs_teacher * lprobs).sum() if reduction == 'sum' else -(probs_teacher * lprobs).sum(-1,
                                                                                                              keepdim=True)
        loss = alpha * guide_loss + (1.0 - alpha) * loss

        mask = target.ne(self.padding_idx)
        correct = torch.sum(logits.argmax(1).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)
        return loss, nll_loss, correct, total, guide_loss

    def step_logging_output(
            self,
            acc,
            nll_loss,
            guide_loss=None,
            reg_cost=None,
    ):
        logging_output = {}
        for k in acc.keys():
            logging_output[f'acc_{k}'] = item(acc[k].data)
            logging_output[f'nll_loss_{k}'] = item(nll_loss[k].data)
        if guide_loss is not None:
            logging_output[f'guide_loss'] = item(guide_loss.data)
        logging_output[f'reg_loss'] = item(reg_cost.data) if reg_cost is not None else 0.0

        return logging_output
