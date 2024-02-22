import os
from pathlib import Path

import datasets
from datasets import ClassLabel, DownloadConfig, DatasetDict
from transformers import DataCollatorForTokenClassification, AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

logger = datasets.logging.get_logger(__name__)

seqeval = evaluate.load("seqeval")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

_CITATION = ""
_DESCRIPTION = """\

"""

class EHRIConfig(datasets.BuilderConfig):
    """BuilderConfig for EHRI"""

    def __init__(self, **kwargs):
        """BuilderConfig for EHRI.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(EHRIConfig, self).__init__(**kwargs)


class EHRI(datasets.GeneratorBasedBuilder):
    """EHRI dataset."""

    BUILDER_CONFIGS = [
        EHRIConfig(name="EHRI", version=datasets.Version("1.0.0"), description="EHRI dataset"),
    ]

    def __init__(self,
                 *args,
                 cache_dir,
                 ner_tags=("B-PERS", "I-PERS", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-GHETTO", "I-GHETTO",
                 "B-CAMP", "I-CAMP", "B-DATE", "I-DATE", "O"),
                 **kwargs):
        self._ner_tags = ner_tags
        super(EHRI, self).__init__(*args, cache_dir=cache_dir, **kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=sorted(list(self._ner_tags))
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": '../dataset/iob/en/ehri_en.txt'}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # EHRI tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }


class EHRI_Dataset(object):
    """
    """
    NAME = "EHRI_Dataset"

    def __init__(self):
        cache_dir = './tmp567dds9'
        print("Cache directory: ", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        download_config = DownloadConfig(cache_dir=cache_dir)
        self._dataset = EHRI(cache_dir=cache_dir)
        print("Cache1 directory: ",  self._dataset.cache_dir)
        self._dataset.download_and_prepare(download_config=download_config)
        self._dataset = self._dataset.as_dataset()

    @property
    def dataset(self):
        return self._dataset

    @property
    def labels(self) -> ClassLabel:
        return self._dataset['train'].features['ner_tags'].feature.names

    @property
    def id2label(self):
        return dict(list(enumerate(self.labels)))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}

    def train(self):
        return self._dataset['train']

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print('\n')
    print(results)
    print('\n')
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


dataset = EHRI_Dataset()
_dataset = dataset.dataset

ds_train_devtest = _dataset['train'].train_test_split(test_size=0.2, seed=42)
ds_devtest = ds_train_devtest['test'].train_test_split(test_size=0.5, seed=42)

ds_splits = DatasetDict({
    'train': ds_train_devtest['train'],
    'valid': ds_devtest['train'],
    'test': ds_devtest['test']
})

print(ds_splits)

print(ds_splits["train"][15])

'''ds_splits = ds_splits.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

label_list = ds_splits["train"].features[f"ner_tags"].feature.names

labels = [ds_splits["train"].features[f"ner_tags"].feature.names]

model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-large", num_labels=13, id2label=dataset.id2label, label2id=dataset.label2id
)

training_args = TrainingArguments(
    output_dir="EHRI_all",
    save_strategy="no",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    # load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_splits["train"],
    eval_dataset=ds_splits["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('./model_2_14')

model = AutoModelForTokenClassification.from_pretrained('./model_2_14')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_splits["train"],
    eval_dataset=ds_splits["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.evaluate()'''