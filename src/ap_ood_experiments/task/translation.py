import os

import torch

from ap_ood_experiments.task import EmbeddingCreator

class TranslationEmbeddingCreator(EmbeddingCreator):
    def __init__(self, model, tokenizer, input_ctx_len, output_ctx_len, src_lang, trg_lang, checkpoint_path=None):
        super(TranslationEmbeddingCreator, self).__init__()
        self.model = model
        print(f'Loading model {checkpoint_path}')
        state_dict = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(state_dict['state_dict'])
        self.tokenizer = tokenizer
        self.input_ctx_len = input_ctx_len
        self.output_ctx_len = output_ctx_len
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.eos_token_id = tokenizer.eos_token_id

    def input_embeddings(self, src_text):
        self.model.eval()
        tokenized = self.tokenizer(
            src_text['translation'][self.src_lang],
            truncation=True,
            padding='max_length',
            max_length=self.input_ctx_len,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        embedded_input = self.model.input_embed(input_ids)
        encoded = self.model.encoder(x=embedded_input, attention_mask=attention_mask).bfloat16()
        return src_text['translation'][self.src_lang], input_ids, encoded, attention_mask

    def _generate(self, batch, sample=False):
        output = [self.tokenizer.pad_token_id] * len(batch['input_ids'])
        output = [[token] for token in output]
        finished = [False] * len(batch['input_ids'])
        mask = [1] * len(batch['input_ids'])
        mask = [[m] for m in mask]
        scores = []

        embedded_input = self.model.input_embed(batch['input_ids'])
        encoded = self.model.encoder(x=embedded_input, attention_mask=batch['attention_mask'])

        for i in range(self.output_ctx_len - 1):
            output_tensor = torch.tensor(output).to(self.device)
            output_mask = torch.ones([output_tensor.shape[0], output_tensor.shape[1]]).to(self.device)

            embedded_output = self.model.output_embed(output_tensor)
            state = self.model.decoder(
                x=embedded_output,
                attention_mask=torch.tensor(mask).to(self.device),
                cross_attention_weights=encoded,
                cross_attention_mask=batch['attention_mask']
            )

            score = self.model.linear_out(state[:, -1])
            scores.append(score.detach())
            if sample:
                next_token = torch.multinomial(torch.nn.functional.softmax(score, dim=-1), num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(score, dim=-1)
            for j, token in enumerate(next_token):
                token = token.item()
                if not finished[j]:
                    mask[j].append(1)
                    output[j].append(token)
                else:
                    mask[j].append(0)
                    output[j].append(self.tokenizer.pad_token_id)
                if token == self.tokenizer.eos_token_id:
                    finished[j] = True
            if all(finished):
                break
        return torch.tensor(output).to(self.device), scores
    
    def output_embeddings_human(self, src_text):
        self.model.eval()
        src_batch = self.tokenizer(src_text['translation'][self.src_lang], truncation=True, padding='max_length', max_length=self.input_ctx_len, return_tensors="pt").to(self.device)
        trg_batch = self.tokenizer(src_text['translation'][self.trg_lang], truncation=True, padding='max_length', max_length=self.output_ctx_len, return_tensors="pt").to(self.device)
        embedded_input = self.model.input_embed(src_batch['input_ids'])
        encoded = self.model.encoder(x=embedded_input, attention_mask=src_batch['attention_mask'])
        embedded_output = self.model.output_embed(trg_batch['input_ids'])
        state = self.model.decoder(x=embedded_output,
                              attention_mask=trg_batch['attention_mask'],
                              cross_attention_weights=encoded,
                              cross_attention_mask=src_batch['attention_mask'])
        mask = trg_batch['attention_mask'].cpu()
        mask[torch.arange(len(mask)), mask.sum(dim=-1) - 1] = 0  # remove eos token from mask
        return src_text['translation'][self.trg_lang], trg_batch['input_ids'].cpu(), state.bfloat16().cpu(), mask.cpu()

    def output_embeddings(self, src_text):
        self.model.eval()
        batch = self.tokenizer(src_text['translation'][self.src_lang], truncation=True, padding='max_length', max_length=self.input_ctx_len, return_tensors="pt").to(self.device)
        generated, _ = self._generate(batch)
        generated_text = self.tokenizer.batch_decode(generated)
        generated_text = [t.split('</s>')[0] for t in generated_text]
        generated_padded = torch.nn.functional.pad(generated, (0, self.output_ctx_len - generated.shape[1]), value=self.tokenizer.pad_token_id)
        if generated.shape[1] > self.output_ctx_len:
            raise ValueError()
        lengths = torch.argmax((generated == self.tokenizer.eos_token_id).int(), dim=-1)
        lengths = torch.where(torch.max(generated == self.tokenizer.eos_token_id, dim=-1)[0], lengths, torch.tensor(generated.shape[1]).to(self.device))
        assert torch.all(lengths > 0)
        len_range = torch.arange(0, generated.shape[1]).reshape(1, -1).to(self.device)
        #mask = (len_range <= lengths.reshape(-1, 1)).int()
        mask = (len_range < lengths.reshape(-1, 1)).int()
        embedded_input = self.model.input_embed(batch['input_ids'])
        encoded = self.model.encoder(x=embedded_input, attention_mask=batch['attention_mask'])
        embedded_output = self.model.output_embed(generated)
        state = self.model.decoder(x=embedded_output,
                              attention_mask=mask,
                              cross_attention_weights=encoded,
                              cross_attention_mask=batch['attention_mask'])

        assert self.output_ctx_len - state.shape[1] >= 0
        state_padded = torch.nn.functional.pad(state, (0, 0, 0, self.output_ctx_len - state.shape[1]), value=0.).bfloat16().cpu()  # pad sequence dimension to output_ctx_len
        mask_padded = torch.nn.functional.pad(mask, (0, self.output_ctx_len - mask.shape[1]), value=0).cpu()  # pad sequence dimension to output_ctx_len
        return generated_text, generated_padded, state_padded, mask_padded

    def lm_head(self):
        return self.model.linear_out

    @torch.no_grad()
    def generate(self, input_ids, masks, n_generations=1):
        self.model.eval()
        input_ids = torch.stack([input_ids] * n_generations, dim=0).to(self.device)
        masks = torch.stack([masks] * n_generations, dim=0).to(self.device)
        batch = {'input_ids': input_ids, 'attention_mask': masks}
        sequences, scores = self._generate(batch, sample=True)
        return sequences, scores
