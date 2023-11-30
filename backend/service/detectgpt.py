import torch
import tqdm
import re
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
import transformers
import time
import functools

segment_size = 100
output_size = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'model/model_best'

class PunctuationGenerator:

    def load_base_model_and_tokenizer(self, name):
        print(f'Loading BASE model {name}...')
        base_model_kwargs = {}
        base_model_kwargs.update(dict(torch_dtype=torch.float16))
        base_model = transformers.LlamaForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=self.cache_dir)
        optional_tok_kwargs = {}
        base_tokenizer = transformers.CodeLlamaTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=self.cache_dir, padding_side='left')
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        return base_model, base_tokenizer
    
    def load_base_model(self):
        print('MOVING BASE MODEL TO GPU...', end='', flush=True)
        start = time.time()
        try:
            self.mask_model.cpu()
        except NameError:
            pass
        except AttributeError:
            pass # for Code LLama
        self.base_model.to(DEVICE)
        print(f'DONE ({time.time() - start:.2f}s)')

    def load_mask_model(self):
        print('MOVING MASK MODEL TO GPU...', end='', flush=True)
        start = time.time()
        self.base_model.cpu() # added
        try:
            self.mask_model.to(DEVICE)
        except AttributeError:
            pass # for Code Llama
        print(f'DONE ({time.time() - start:.2f}s)')


    def __init__(self):
        print('*setting model*')
        # loading base model
        self.cache_dir = './.cache'
        self.base_model, self.base_tokenizer = self.load_base_model_and_tokenizer("codellama/CodeLlama-7b-hf")

        self.pattern = re.compile(r"<extra_id_\d+>")
        
        # loading mask model
        int8_kwargs = {}
        half_kwargs = {}
        '''
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
            print('using int8')
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
            print('using half')
        '''
        print(f'Loading mask filling model Salesforce/codet5p-770m...')
        self.mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-770m", **int8_kwargs, **half_kwargs, cache_dir=self.cache_dir)
        try:
            n_positions = self.mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
        self.mask_tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codet5p-770m", model_max_length=n_positions, cache_dir=self.cache_dir)
    
    def tokenize_and_mask(self, text, span_length, pct, ceil_pct=False):

        buffer_size = 1

        tokens = text.split(' ')
        mask_string = '<<<mask>>>'

        n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        #FIXED
        if n_spans == 0:
            if len(tokens) - span_length <= 0:
                n_spans = 0
            elif len(tokens) > 10:
                n_spans = 2
            else:
                n_spans = 1

        n_masks = 0
        while n_masks < n_spans:
            # print(n_masks, n_spans, len(tokens))
            start = np.random.randint(0, len(tokens) - span_length)
            end = start + span_length
            search_start = max(0, start - buffer_size)
            search_end = min(len(tokens), end + buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)

        return text

    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]
    
    def replace_masks(self, texts):
        n_expected = self.count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(texts, truncation=True, return_tensors="pt", padding=True).to(DEVICE)
        outputs = self.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=1, num_return_sequences=1, eos_token_id=stop_id)
        result = self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
        return result
    
    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
        # return the text in between each matched mask token
        extracted_fills = [self.pattern.split(x)[1:-1] for x in texts]
        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
        return extracted_fills
    
    def apply_extracted_fills(self, masked_texts, extracted_fills, skip=False):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]
        n_expected = self.count_masks(masked_texts)
        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                if skip == True:
                    print(len(fills))
                    for fill_idx in range(n):
                        if (len(fills) - 1) < fill_idx:
                            print(fill_idx)
                            fill_word = ''
                        else:
                            fill_word = fills[fill_idx]
                        text[text.index(f"<extra_id_{fill_idx}>")] = fill_word
                else:
                    tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts

    def perturb_texts(self, texts, span_length, pct, ceil_pct=False):
        chunk_size = 10
        outputs = []
        for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
            outputs.extend(self.perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
        return outputs

    def perturb_texts_(self, texts, span_length, pct, ceil_pct=False):
        masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            skip = False
            if attempts > 3:
                skip = True
            
            new_perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills, skip)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
        return perturbed_texts
    
    # Get the log likelihood of each text under the base_model
    def get_ll(self, text):
        with torch.no_grad():
            tokenized = self.base_tokenizer(text,  truncation=True, return_tensors="pt").to(DEVICE) # FIXED
            labels = tokenized.input_ids
            return -self.base_model(**tokenized, labels=labels).loss.item()
    
    def get_lls(self, texts):
        return [self.get_ll(text) for text in texts]
    
    def run_perturbation_experiment(self, results, criterion, span_length=10, n_perturbations=1, n_samples=500):
        # compute diffs with perturbed
        predictions = {'real': []}
        for res in results:
            if criterion == 'd':
                predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            elif criterion == 'z':
                if res['perturbed_original_ll_std'] == 0:
                    res['perturbed_original_ll_std'] = 1
                    print("WARNING: std of perturbed original is 0, setting to 1")
                    print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                    print(f"Original text: {res['original']}")
                predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
        
        #FIXME: for NaN inputs
        last_predictions_real = 0
        for i in range(0, len(predictions['real'])):
            if np.isnan(predictions['real'][i]):
                predictions['real'][i] = last_predictions_real
                print('isNaN')
            else:
                last_predictions_real = predictions['real'][i]

        threshold = 0.0

        
        if predictions['real'][0] > threshold:
            return 'human'
        else:
            return 'machine'

    def get_perturbation_results(self, span_length=10, n_perturbations=1, n_samples=500):
        self.load_mask_model()

        torch.manual_seed(0)
        np.random.seed(0)
        results = []
        original_text = [self.text]

        perturb_fn = functools.partial(self.perturb_texts, span_length=span_length, pct=0.3)
        print('perturbing original text')
        p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])

        try:
            p_original_text = perturb_fn(p_original_text)
        except AssertionError:
            pass

        assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
            })

        self.load_base_model()

        for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
            p_original_ll = self.get_lls(res["perturbed_original"])
            res["original_ll"] = self.get_ll(res["original"])
            res["all_perturbed_original_ll"] = p_original_ll
            res["perturbed_original_ll"] = np.mean(p_original_ll)
            res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

        return results


    async def generate(self, text):
        self.text = text
        perturbation_results = self.get_perturbation_results(2, 3, 1)
        output = self.run_perturbation_experiment(perturbation_results, 'd', span_length=2, n_perturbations=3, n_samples=1)
        return {
            'result': perturbation_results,
            'label': output
        }