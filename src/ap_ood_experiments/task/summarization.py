import torch

from ap_ood_experiments.task import EmbeddingCreator

class SummarizationEmbeddingCreator(EmbeddingCreator):
    def __init__(self, model, tokenizer, input_ctx_len, output_ctx_len):
        super(SummarizationEmbeddingCreator, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.input_ctx_len = input_ctx_len
        self.output_ctx_len = output_ctx_len
        self.eos_token_id = tokenizer.eos_token_id

    def input_embeddings(self, src_text):
        self.model.eval()
        self.model.model.eval()
        self.model.model.encoder.eval()
        batch = self.tokenizer(src_text['document'], truncation=True, padding='max_length', max_length=self.input_ctx_len, return_tensors="pt").to(self.device)
        return src_text['document'], batch['input_ids'], self.model.model.encoder(batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state, batch['attention_mask']
 
    def output_embeddings_human(self, src_text):
        self.model.eval()
        self.model.model.eval()
        self.model.model.encoder.eval()
        self.model.model.decoder.eval()
        src_batch = self.tokenizer(src_text['document'], truncation=True, padding='max_length', max_length=self.input_ctx_len, return_tensors="pt").to(self.device)
        encoded = self.model.model.encoder(**src_batch).last_hidden_state
        trg_batch = self.tokenizer(src_text['summary'], truncation=True, padding='max_length', max_length=self.output_ctx_len, return_tensors="pt").to(self.device)
        decoded = self.model.model.decoder(input_ids=trg_batch['input_ids'], attention_mask=trg_batch['attention_mask'], encoder_hidden_states=encoded, encoder_attention_mask=src_batch['attention_mask']).last_hidden_state
        mask = trg_batch['attention_mask']
        mask[torch.arange(mask.shape[0]), mask.sum(dim=-1) - 1] = 0  # set eos token mask to 0
        return src_text['summary'], trg_batch['input_ids'].cpu(), decoded.cpu(), mask.cpu()

    def output_embeddings(self, src_text):
        self.model.eval()
        batch = self.tokenizer(src_text['document'], truncation=True, padding='max_length', max_length=self.input_ctx_len, return_tensors="pt").to(self.device)
        generated = self.model.generate(**batch, max_new_tokens=self.output_ctx_len - 1, do_sample=False)  # max_new_tokens ignores the padding token at the start of the generation
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
        encoded = self.model.model.encoder(**batch).last_hidden_state
        decoded = self.model.model.decoder(input_ids=generated, attention_mask=mask, encoder_hidden_states=encoded, encoder_attention_mask=batch['attention_mask'])
        state = decoded.last_hidden_state
        assert self.output_ctx_len - state.shape[1] >= 0
        state_padded = torch.nn.functional.pad(state, (0, 0, 0, self.output_ctx_len - state.shape[1]), value=0.).cpu()  # pad sequence dimension to output_ctx_len
        mask_padded = torch.nn.functional.pad(mask, (0, self.output_ctx_len - mask.shape[1]), value=0).cpu()  # pad sequence dimension to output_ctx_len
        return generated_text, generated_padded, state_padded, mask_padded

    def lm_head(self):
        return self.model.lm_head

    def generate(self, input_ids, masks, n_generations=1):
        self.model.eval()
        batch = {'input_ids': input_ids.unsqueeze(0).to(self.device), 'attention_mask': masks.unsqueeze(0).to(self.device)}
        generated = self.model.generate(
            **batch,
            max_new_tokens=self.output_ctx_len - 1,
            num_return_sequences=n_generations,
            do_sample=True,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=None,
        )
        return generated.sequences, generated.scores
