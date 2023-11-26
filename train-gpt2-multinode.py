import torch
from accelerate import Accelerator

from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import set_seed, GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import DataLoader, random_split

set_seed(731)

import os
import pandas as pd
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.train = train
        self.data = pd.read_csv(os.path.join('./data', 'train.csv' if train else 'test.csv'))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        record = self.data.iloc[index]
        text = record['text']
        if self.train:
            return {'text': text, 'label': record['target']}
        else:
            return {'text': text, 'label': '0'}

class Gpt2ClassificationCollator(object):
    def __init__(self, tokenizer, max_seq_len=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        return
    
    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [int(sequence['label']) for sequence in sequences]
        inputs = self.tokenizer(text=texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.max_seq_len)
        inputs.update({'labels': torch.tensor(labels)})
        
        return inputs

def main():
      model_config = GPT2Config.from_pretrained('gpt2', num_labels=2) # Binary Classification
      model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=model_config)

      tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
      tokenizer.padding_side = "left" # Very Important
      tokenizer.pad_token = tokenizer.eos_token

      model.resize_token_embeddings(len(tokenizer))
      model.config.pad_token_id = model.config.eos_token_id

      gpt2classificationcollator = Gpt2ClassificationCollator(tokenizer=tokenizer,
                                                        max_seq_len=60)

      train_dataset = TweetDataset(train=True)
      test_dataset = TweetDataset(train=False)

      for i in range(10):
          print(train_dataset.__getitem__(i)['text'])


      train_size = int(len(train_dataset) * 0.8)
      val_size = len(train_dataset) - train_size
      train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

      training_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=True,
                              collate_fn=gpt2classificationcollator)
      val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=32,
                            shuffle=False,
                            collate_fn=gpt2classificationcollator)
      test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=32,
                             shuffle=False,
                             collate_fn=gpt2classificationcollator)


      total_epochs = 10

      param_optimizer = list(model.named_parameters())
      no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
      optimizer_grouped_parameters = [
          {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
          {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]
      optimizer = AdamW(optimizer_grouped_parameters,
                  lr=1e-5,
                  eps=1e-8)

      num_train_steps = len(training_dataloader) * total_epochs
      num_warmup_steps = int(num_train_steps * 0.1) 

      scheduler = get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=num_warmup_steps,
                                               num_training_steps = num_train_steps)

      accelerator = Accelerator()

      model, optimizer, training_dataloader, scheduler = accelerator.prepare(
          model, optimizer, training_dataloader, scheduler
      )
      prediction_labels = []
      true_labels = []
      total_loss = []
      for batch in training_dataloader:
         true_labels += batch['labels'].cpu().numpy().flatten().tolist()
         batch = {k:v.type(torch.long).to("cuda") for k, v in batch.items()}
         outputs = model(**batch)
         loss, logits = outputs[:2]
         logits = logits.detach().cpu().numpy()
         total_loss.append(loss.item())
         optimizer.zero_grad()
         accelerator.backward(loss)
         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent exploding gradient

         optimizer.step()
         scheduler.step()
         prediction_labels += logits.argmax(axis=-1).flatten().tolist()

if __name__ == "__main__":
    main()
