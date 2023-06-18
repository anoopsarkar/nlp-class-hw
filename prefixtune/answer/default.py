import argparse, os, string, sys
import torch
import sacrebleu
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
# import peft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TableToText:

    def __init__(
            self,
            modelfile,
            modelsuffix='.pt',
            basemodel='distilgpt2',
            traindata='e2e_nlg_cleaned',
            epochs=5,
            batchsize=4,
            lr=5e-5,
            virtualtokens=5,
            prefixprojection=False
        ):
        # the input sentences will be handled using this object, you do not need to manually encode input sentence words
        self.tokenizer = AutoTokenizer.from_pretrained(basemodel)
        self.tokenizer_pad_token_id = self.tokenizer.eos_token_id \
            if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        self.traindata = traindata
        self.modelfile = modelfile
        self.modelsuffix = modelsuffix
        self.basemodel = basemodel
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr
        self.virtualtokens = virtualtokens
        self.prefixprojection = prefixprojection
        self.prompt = "Convert the following table into English text: "
        self.training_data = []
        self.model = None # setup the model in self.decode() or self.train()

    def preprocess_function(self, examples):
        text_column = "meaning_representation"
        label_column = "human_reference"
        max_length = 150
        batch_size = len(examples[text_column])
        inputs = [f"{self.prompt}{x} {self.tokenizer.bos_token} " for x in examples[text_column]]
        targets = [f"{x} {self.tokenizer.eos_token}" for x in examples[label_column]]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer_pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer_pad_token_id] * (
                    max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = \
                [0] * \
                (max_length - len(sample_input_ids)) + \
                model_inputs["attention_mask"][i]
            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def get_data(self, splits=("train", )):
        """
        Loads the requested dataset with name == :param dataset_name: and returns dataloaders over each split defined
          in :param splits: which can contain any subset of ("train", "validation", "test"). The dataloder batchsize will be
            defined using :param self.batchsize:.
        """
        dataset = load_dataset(self.traindata)
        processed_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset"
        )

        data_loaders = {}
        for split in splits:
            assert split in processed_datasets
            data_loaders[split] = DataLoader(
                                    processed_datasets[split],
                                    collate_fn=default_data_collator,
                                    batch_size=self.batchsize,
                                    pin_memory=True,
                                    shuffle=(split == "train")
                                  )
        return data_loaders

    def train(self):
        data_loaders = self.get_data(splits=("train", ))
        model = AutoModelForCausalLM.from_pretrained(self.basemodel)

        # You can print the parameters for debugging or understanding the code
        # but make sure you comment it out otherwise it will pollute the output
        # that is produced for dev and test
        #model.print_trainable_parameters()

        # TODO
        # if using HF peft module, then add calls to PrefixTuningConfig and get_peft_model
        # which will take num_virtual_tokens which is set to self.virtualtokens and
        # prefix_projection which is set to self.prefixprojection

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(data_loaders["train"]) * self.epochs),
        )
        model = model.to(device)

        for epoch in range(self.epochs):
            model.train()

            # TODO rest of the training steps for prefix tuning

            if epoch == self.epochs - 1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + epoch_str + self.modelsuffix
            model.save_pretrained(savefile)

    def decode(self, model, inputfile):
        inputpath = Path(inputfile)
        assert inputpath.exists()
        with inputpath.open() as f:
            lines = [line for line in f.read().splitlines() if len(line) > 0 and not line.isspace()]
            decoder_output = []
            for i, src in tqdm(enumerate(lines)):
                predicted_line = self.predict(model, src, num_sequences=1)
                #if not predicted_line or src.split()[0] not in predicted_line.split():
                    # if output generation failed then use a heuristic to generate some output
                    #predicted_line = src.replace(':', '').replace('|', '').replace('  ', ' ')
                decoder_output.append(f"{i}||{predicted_line}")
        return decoder_output

    def predict(self, model, src, num_sequences=1):
        inputs = self.tokenizer(self.prompt + src + ' ' + self.tokenizer.bos_token + ' ', return_tensors="pt")
        prediction = None
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer_pad_token_id,
                do_sample=True,
                num_beams=5,
                top_p=0.9,
                temperature=1.0,
                num_return_sequences=num_sequences
            )
            # TODO you may want to generate more than one sequence and choose the best one!
            text = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
            return text.lstrip().replace(self.prompt + src, "").replace("\n", " ")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", dest="inputfile",
                            default=os.path.join('data', 'input', 'dev.txt'),
                             help="produce table to text output for these input tables")
    argparser.add_argument("-t", "--traindata", dest="traindata",
                            default='e2e_nlg_cleaned',
                            help="name of hugging face cleaned up dataset for the E2E table to text task")
    argparser.add_argument("-v", "--virtualtokens", dest="virtualtokens",
                            type=bool, default=5,
                            help="number of virtual prompt tokens for prefix tuning")
    argparser.add_argument("-p", "--prefixprojection", dest="prefixprojection",
                            action="store_true", default=False,
                            help="whether to project the prefix embeddings")
    argparser.add_argument("-m", "--modelfile", dest="modelfile",
                            default=os.path.join('data', 'peft'),
                            help="filename without suffix for model files")
    argparser.add_argument("-s", "--modelsuffix", dest="modelsuffix", default='.pt',
                            help="filename suffix for model files")
    argparser.add_argument("-M", "--basemodel", dest="basemodel",
                            default='distilgpt2',
                            help="The base huggingface pretrained model to be used as the encoder.")
    argparser.add_argument("-e", "--epochs", dest="epochs", type=int, default=1,
                            help="number of epochs [default: 1]")
    argparser.add_argument("-b", "--batchsize", dest="batchsize", type=int, default=16,
                            help="batch size [default: 16]")
    argparser.add_argument("-r", "--lr", dest="lr", type=float, default=5e-5,
                            help="the learning rate used to finetune the BERT-like encoder module.")
    argparser.add_argument("-f", "--force", dest="force", action="store_true", default=False,
                            help="force training phase (warning: can be slow)")
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None,
                            help="log file for debugging")
    opts = argparser.parse_args()
    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)
    modelfile = opts.modelfile
    if modelfile.endswith('.pt'):
        modelfile = modelfile.removesuffix('.pt')
    table_to_text = TableToText(
                        modelfile,
                        modelsuffix=opts.modelsuffix,
                        basemodel=opts.basemodel,
                        traindata=opts.traindata,
                        epochs=opts.epochs,
                        batchsize=opts.batchsize,
                        lr=opts.lr,
                        virtualtokens=opts.virtualtokens,
                        prefixprojection=opts.prefixprojection
                    )
    # TODO default.py always uses a prompt to produce output from the pretrained model
    # when you have implemented prefix tuning then change this to False to train and/or 
    # use your prefix tuned model
    model = None
    if True:
        print(f"Loading the non-finetuned pre-trained model: {opts.basemodel}", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(opts.basemodel)
        model = model.to(device)
    else:
        if not os.path.isdir(modelfile + opts.modelsuffix) or opts.force:
            print(f"Could not find modelfile {modelfile + opts.modelsuffix} or -f used. Starting training.", file=sys.stderr)
            table_to_text.train()
            print("Training done.", file=sys.stderr)
        # use the model file if available and opts.force is False
        assert(os.path.isdir(modelfile + opts.modelsuffix))
        print(f"Found modelfile {modelfile + opts.modelsuffix}. Starting decoding.", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(opts.basemodel)
        # TODO: if using hf peft library for prefix tuning:
        # model = PeftModel.from_pretrained(model, modelfile + opts.modelsuffix)
        model = model.to(device)
    if model:
        decoder_output = table_to_text.decode(model, opts.inputfile)
        print("\n".join(decoder_output))
