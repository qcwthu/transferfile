import os
import pdb
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

class T5forAll(nn.Module):
    def __init__(self, args, model, tokenizer):
        super(T5forAll, self).__init__()
        self.args = args
        self.model = model
        ### load ckpt
        if args.use_lm_adapted == 1:
            print("use lm adapted model!")
            t5ckpt = torch.load(args.lm_adapted_path)
            if args.ifckpt_onlymodel == 1:
                self.model.load_state_dict(t5ckpt)
            else:
                self.model.load_state_dict(t5ckpt['t5-large-prefixlm'])
            ### if prompt tuning, set requires_grad false
            for name, param in self.model.named_parameters():
                #print(name)
                param.requires_grad = False
        self.tokenizer = tokenizer
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id
        self.promptnumber = 0
        self.promptembedding = None

    def set_prompt_embedding(self,promptnumber,promptembedding):
        self.promptnumber = promptnumber
        self.promptembedding = nn.parameter.Parameter(promptembedding)
        # print(self.promptnumber)
        # print(self.promptembedding.shape)
        # print(self.promptembedding.requires_grad)

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        ##### handle prompt, cal input_embed
        input_embed_part = self.model.encoder.embed_tokens(input_ids)
        #print(input_embed_part.shape)
        #print(self.promptembedding)
        prompt_embed_repeat = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        #print(prompt_embed_repeat.shape)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        #print(allembedding.shape)
        #print(attention_mask.shape)
        mask_prompt = torch.full((attention_mask.shape[0],self.promptnumber),1).to(self.args.device)
        all_attention_mask = torch.cat([attention_mask, mask_prompt], 1)
        #print(all_attention_mask.shape)
        return self.model(
            inputs_embeds=allembedding,
            attention_mask=all_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        # return self.model(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     decoder_input_ids=decoder_input_ids,
        #     decoder_attention_mask=decoder_attention_mask,
        #     labels=labels
        # )

    def forward(self, batch):
        lm_labels = batch["target_ids"]
        #print(self.tokenizer.pad_token_id)
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        #print(self.model.config.decoder_start_token_id)
        #print(self.model.config.bos_token_id)
        outputs = self._step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0]

        return loss

    def _generative_step(self, batch):
        input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        #print(input_embed_part.shape)
        prompt_embed_repeat = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        #print(prompt_embed_repeat.shape)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        #print(allembedding.shape)
        #print(batch["attention_mask"].shape)
        mask_prompt = torch.full((batch["attention_mask"].shape[0], self.promptnumber), 1).to(self.args.device)
        #print(mask_prompt.shape)
        all_attention_mask = torch.cat([batch["attention_mask"], mask_prompt], 1)
        #print(all_attention_mask.shape)
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )
        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            #decoder_attention_mask=batch['target_mask'],
            #max_length=self.args.max_length,
            max_length=128,  ####128 max summarization length
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        # generated_ids = self.model.generate(
        #     inputs_embeds=allembedding,
        #     decoder_input_ids=decoder_input_ids,
        #     attention_mask=all_attention_mask,
        #     use_cache=True,
        #     # decoder_attention_mask=batch['target_mask'],
        #     max_length=self.args.max_length,
        #     do_sample=True,
        #     repetition_penalty=2.5,
        #     length_penalty=1.0,
        #     early_stopping=True,
        #     top_k = 64,
        #     num_return_sequences=4
        # )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
        input = self.ids_to_clean_text(batch["input_ids"])
        return input,target,preds

    def _generative_samples(self, batch):
        input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        #print(input_embed_part.shape)
        prompt_embed_repeat = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        #print(prompt_embed_repeat.shape)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        #print(allembedding.shape)
        #print(batch["attention_mask"].shape)
        mask_prompt = torch.full((batch["attention_mask"].shape[0], self.promptnumber), 1).to(self.args.device)
        #print(mask_prompt.shape)
        all_attention_mask = torch.cat([batch["attention_mask"], mask_prompt], 1)
        #print(all_attention_mask.shape)
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )

        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            max_length=self.args.max_length,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=True,
            top_k = 64,
            #top_p = 0.85,
            num_return_sequences=3
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
        input = self.ids_to_clean_text(batch["input_ids"])
        return input,target,preds


    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return self.lmap(str.strip, gen_text)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
