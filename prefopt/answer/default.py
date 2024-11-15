import torch
import argparse
import sys
import logging
import os
import json
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

def decode_all(model, device, inputfile):
    pipe = pipeline(
        "text-generation",
        device=device,
        model=model,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
        }
    )

    text = Path(inputfile).read_text().strip().split('\n')
    for line in tqdm(text, total=len(text)):
        line = line.strip()
        data = json.loads(line)
        prompt_text = data['prompt'] + '\n' + data['constraints']
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides useful answers without too much extra output.",
            },
            {
                "role": "user", 
                "content": f"{prompt_text}"
            },
        ]
        outputs = pipe(
            messages,
            do_sample=True,
            temperature=0.3,
            max_new_tokens=128,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        if outputs:
            print(json.dumps({'output': outputs[0]["generated_text"][-1]["content"]}))
        else:
            print(json.dumps({'output': "Sorry!"}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate answers from an LLM")

    parser.add_argument("-m", "--model", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="model path or model id")
    parser.add_argument("-i", "--inputfile",
                        default=os.path.join('data', 'input', 'dev.txt'),
                        help="produce output for this input file")
    parser.add_argument("-d", "--device",
                        default='cpu',
                        help="cuda device if available")
    parser.add_argument("-l", "--logfile", dest="logfile", default=None,
                        help="log file for debugging")
    args = parser.parse_args()
    assert(args.model)
    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile, filemode='w', level=logging.DEBUG)
    device = 'cpu'
    device = args.device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"using device {args.device}", file=sys.stderr)
    decode_all(args.model, device, args.inputfile)
