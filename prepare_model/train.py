from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

columns = {0: "text", 1: "is_sponsor"}
data_folder = "."

print("Creating label dictionary...")
corpus: Corpus = ColumnCorpus(
    data_folder, columns, train_file="head.txt", in_memory=False
)
label_dict = corpus.make_label_dictionary(label_type=columns[1])

print("Loading corpus...")
# eval empty because flair broke when evaluating model
corpus: Corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="data.txt",
    in_memory=False,
    test_file="empty.txt",
    dev_file="empty.txt",
)

embeddings = TransformerWordEmbeddings(
    model="microsoft/xtremedistil-l12-h384-uncased",
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=64,
)

tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type=columns[1],
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False,
)

trainer = ModelTrainer(tagger, corpus)

trainer.fine_tune(
    "sponsor-flert",
    learning_rate=5e-5,
    mini_batch_size=12,
    mini_batch_chunk_size=1,
    max_epochs=1,
)
