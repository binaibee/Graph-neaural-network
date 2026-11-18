# Project Handbook

**Scam Detection Using Graph Neural Networks Across Multi-Channel Communications**

This handbook explains the whole project from beginning to end. It is written for you, the author, so that you can
understand every part and defend it. Everything in it describes what exists in this folder. Every number comes from
`results/results.json`, which is produced by running `python -m scam_gnn.train`. Where something is planned but not
built, it is labelled **future work**.

How to read it: Parts 1 to 3 give the big picture, Parts 4 to 12 follow the pipeline stage by stage, Parts 13 to 17
cover the demo, testing, limitations and history, Part 18 is the call-data decision, and Part 20 is defence preparation.

---

## 1. The problem

Scammers reach people through several channels: text messages, emails and web links. A single scam campaign usually
reuses the same things: the same phone number to call back, the same website, the same sender address.

Most detection systems look at **one message at a time** and ask "does this text look like a scam?". That ignores the
reuse. If ten messages all point to the same website and nine of them are known scams, the tenth deserves suspicion
even if its wording is harmless.

**The research question of this project:** does representing communications and the things they share as a *graph*,
and classifying them with a *Graph Neural Network*, detect scams better than standard models that look at each message
alone?

## 2. Objectives

| # | Objective | Where it is met |
|---|---|---|
| 1 | Combine several communication channels into one dataset format | `data.py` |
| 2 | Represent messages and what they share as a graph | `graph.py` |
| 3 | Train a GNN to classify messages as scam or legitimate | `models.py`, `train.py` |
| 4 | Compare the GNN fairly with non-graph models | `train.py`, `evaluate.py` |
| 5 | Measure whether the graph itself is what helps (ablation) | `train.py` |
| 6 | Score a new message on demand and explain the result | `predict.py`, `explain.py` |
| 7 | Demonstrate the system | `app/app.py` |

## 3. The system at a glance

```
Problem -> Data -> Preprocessing -> Graph construction -> Features -> Model -> Prediction -> Explanation -> Evaluation -> Demonstration
           data.py                  graph.py              features.py  models.py  predict.py   explain.py     evaluate.py   app/app.py
                                                                       train.py
```

There are two ways the system runs:

- **Training run** (`python -m scam_gnn.train`, about 14 minutes): executes stages 1 to 6 on all the data and saves the
  trained GNN and all results into `results/`.
- **Demonstration** (`run_demo.bat`): loads the saved GNN and graph, and performs stages 7 and 8 on one message you
  type. It takes about half a second per message.

### Files and folders

| Path | Purpose |
|---|---|
| `config.json` | Every setting: file paths, sample sizes, seeds, training settings |
| `data/public/` | The three datasets as CSV files |
| `src/scam_gnn/config.py` | Reads `config.json` |
| `src/scam_gnn/normalize.py` | Small text helpers: find links, phone numbers and email addresses |
| `src/scam_gnn/data.py` | Stage 1: load, clean, de-duplicate, split |
| `src/scam_gnn/graph.py` | Stage 2: build nodes and edges |
| `src/scam_gnn/features.py` | Stage 3: turn nodes into numbers |
| `src/scam_gnn/models.py` | Stage 4: the two baselines and the GNN |
| `src/scam_gnn/train.py` | Stage 5: train, run the ablation, save everything |
| `src/scam_gnn/evaluate.py` | Stage 6: metrics |
| `src/scam_gnn/predict.py` | Stage 7: score one new message |
| `src/scam_gnn/explain.py` | Stage 8: explain a score |
| `src/scam_gnn/download_model.py` | Optional helper for the MiniLM text model (not used by default) |
| `app/app.py`, `app/graph_view.py` | The demonstration page and the graph picture |
| `app/test_cases.json` | 17 hand-written test inputs |
| `results/results.json` | All results |
| `results/model.pt` | The trained GNN |
| `results/prepared.pkl` | The stored graph and features, so the demo starts fast |
| `results/predictions.csv` | The risk score of every message from each model |
| `tests/` | 15 automated checks |
| `_archive/` | Files of the earlier version that are no longer used |

---

## 4. Stage 1: the data

### 4.1 The three datasets

| Channel | Dataset | What one record is | Loaded | Used after cleaning | Scam | Legitimate |
|---|---|---|---|---|---|---|
| SMS | UCI SMS Spam Collection | One text message, labelled spam or ham | 5,574 | 5,159 | 642 | 4,517 |
| Email | SpamAssassin public corpus | One email with sender, receiver, subject, body, labelled spam or ham | 5,809 | 5,784 | 1,694 | 4,090 |
| Web link | PhiUSIIL Phishing URL dataset | One URL, labelled phishing or legitimate | 15,000 (sampled from 235,795) | 14,996 | 6,418 | 8,578 |
| **Total** | | | | **25,939** | 8,754 | 17,185 |

Sources and nature:

- **UCI SMS Spam Collection.** A public research dataset of real English SMS messages, mostly from the UK and
  Singapore, collected around 2011 to 2012. It contains only the text and the label. There is **no sender, no
  recipient and no time**.
- **SpamAssassin public corpus.** Real emails from about 2002, published by the Apache SpamAssassin project for testing
  spam filters. This project uses the CSV version from the Zenodo collection "Phishing Email Curated Datasets"
  (DOI 10.5281/zenodo.8339691, licence CC BY 4.0). It has the **sender address (on 99.8% of emails) and receiver
  address (on 95.3%)**. This is the only one of the three datasets with real relationship information.
- **PhiUSIIL.** A public dataset of 235,795 URLs labelled phishing or legitimate. The original file also contains about
  50 ready-made columns describing each web page. **This project uses only the URL and its label**, for the reasons in
  4.4.

All three are "weakly" matched to the word *scam*: the SMS and email datasets are labelled *spam* (unwanted bulk
messages, a large share of which are scams) and the URL dataset is labelled *phishing*. In this project label 1 means
"spam, scam or phishing" and label 0 means "legitimate". Be ready to say this precisely.

### 4.2 One common format

`data.py` converts every record into one row of a table called **events**:

| Column | Meaning |
|---|---|
| `event_id` | A unique name such as `sms-12` or `email-340` |
| `channel` | `sms`, `email` or `url` |
| `label` | 1 = scam, 0 = legitimate |
| `text` | The SMS text, the email subject and body, or the URL |
| `sender` | The sender's email address (emails only; empty otherwise) |
| `recipient` | The recipient's email address (emails only; empty otherwise) |
| `split` | `train`, `validation` or `test` |

*Why one format?* So that every later stage treats all channels the same way. This is what makes the system
"multi-channel": one graph, one feature set, one model.

### 4.3 Preprocessing steps (in order)

1. **Clean the text.** Runs of spaces, tabs and line breaks become one space. URLs are put in a standard form
   (lower-case website name, no trailing slash).
2. **Shorten very long emails** to the first 2,000 characters.
3. **Sample.** PhiUSIIL is far larger than the other two, so a random sample of 15,000 URLs is taken that keeps the
   original phishing rate. (*Why?* So one channel does not swamp the others and so training fits on a laptop.)
4. **Remove duplicates.** If the same text appears twice in a channel, only the first copy is kept: 415 SMS, 25 emails
   and 4 URLs were removed. (*Why?* Otherwise a message could be in the training data and, word for word, in the test
   data. The model would be "tested" on something it had memorised. This is called **data leakage**. In the earlier
   version of the project, 91 test SMS were also in the training set.)
5. **Split** into training (60%), validation (20%) and test (20%).

### 4.4 Why only the URL text is used from PhiUSIIL

The original PhiUSIIL file includes columns such as number of lines of page code, number of images and a "similarity
index". Two problems:

- They make the task trivially easy. In the earlier version, one column alone separated the classes about 99.5% of the
  time, so every model scored close to 100% and nothing could be learned from the comparison.
- They cannot be computed for a new link without downloading and analysing the web page, so a link typed into the demo
  could not be scored honestly.

Using only the URL itself solves both: the task is realistic, and all three channels can be typed into the demo.

### 4.5 How the split works

For each combination of channel and label (for example "SMS scams"), the records are shuffled with a fixed random seed
(`data_seed: 7`) and divided 60/20/20. This is called a **stratified random split**. It guarantees every channel has
the same scam rate in all three parts, and the fixed seed makes it reproducible.

| | Train | Validation | Test |
|---|---|---|---|
| SMS | 3,095 | 1,031 | 1,033 |
| Email | 3,470 | 1,157 | 1,157 |
| Web link | 8,996 | 3,000 | 3,000 |
| **Total** | 15,561 | 5,188 | 5,190 |

*Why not split by time?* Two of the three datasets have no timestamps at all. The earlier version invented timestamps
and called the split "time-based", which was misleading. A random split is the honest choice. Its weakness is that it
does not test whether the model copes with scams that appear *after* training (see limitations).

---

## 5. Stage 2: graph construction

### 5.1 What a graph is

A graph is a set of **nodes** (dots) joined by **edges** (lines). Here the graph has several *types* of node and
several *types* of edge, which is called a **heterogeneous graph**.

### 5.2 Nodes

| Node type | What it is | Count |
|---|---|---|
| `message` | One SMS, email or URL record. This is what gets classified. | 25,939 |
| `url` | A link found inside a message (or the URL record itself) | 18,719 |
| `domain` | A website name such as `www.example.com`, or the mail domain of a sender | 17,386 |
| `sender` | An email sender address | 2,409 |
| `phone` | A phone number found in the text of a message | 1,253 |
| `recipient` | An email recipient address | 1,182 |
| **Total** | | **66,888** |

### 5.3 Edges

| Edge type | From → to | Meaning | Count |
|---|---|---|---|
| `sent_by` | message → sender | Who sent the email | 5,774 |
| `sent_to` | message → recipient | Who received the email | 5,514 |
| `contains_url` | message → url | The message contains this link | 23,535 |
| `contains_phone` | message → phone | The message contains this phone number | 1,857 |
| `resolves_to_domain` | url → domain | The link is on this website | 18,718 |
| `uses_domain` | sender → domain | The sender's address is at this mail domain | 2,409 |
| `shares_sender` | message ↔ message | Two emails from the same sender | 71,612 |
| `shares_recipient` | message ↔ message | Two emails to the same recipient | 98,550 |
| `shares_infrastructure` | message ↔ message | Two messages with the same phone number or the same website | 96,260 |
| **Total** | | | **324,229** |

### 5.4 How it is built (`graph.py`)

For every event:

1. Create a `message` node.
2. If there is a sender, create a `sender` node and a `sent_by` edge; also link the sender to its mail domain.
3. If there is a recipient, create a `recipient` node and a `sent_to` edge.
4. Find every link in the text (at most 10 per message). For each, create a `url` node, a `contains_url` edge, and a
   `resolves_to_domain` edge to the website.
5. Find every phone number in the text (9 to 15 digits). For each, create a `phone` node and a `contains_phone` edge.
6. After all events are processed, connect messages that share a sender, a recipient, a phone number or a website with
   a direct message-to-message edge.

**The group-size limit.** If more than 128 messages share the same thing (for example a mailing-list address that
receives hundreds of emails, or a very popular website), no message-to-message edges are created for that group. Two
reasons: such a shared item says little about any one message, and connecting every pair would create millions of
edges. Six recipient addresses, covering 2,504 emails, are over the limit. Those emails are still connected to the
recipient *node*.

### 5.5 How the channels are represented and connected

All channels live in the same graph and use the same node and edge types. A channel is just a property of the message
node (and one of the model's input features).

How connected each channel is:

| Channel | Messages | Linked to at least one other message | Scam rate among linked | Scam rate among unlinked |
|---|---|---|---|---|
| Email | 5,784 | 4,623 (80%) | 21% | 62% |
| SMS | 5,159 | 174 (3%) | 100% | 9% |
| Web link | 14,996 | 541 (4%) | 99.6% | 41% |

What this table tells you, and you should be able to say it:

- **Email is richly connected** because it has real senders and recipients. Linked emails are mostly legitimate
  (mailing-list traffic between regular correspondents), unlinked ones mostly spam (one-off senders).
- **SMS and web links are sparsely connected** because those datasets have no sender information. The only links come
  from phone numbers and websites that appear in more than one message, and those are almost always scams.
- **Between channels there are only 8 edges**, all between an email and a URL record that point to the same website.
  The three datasets were collected separately, in different years and countries, so they share almost nothing. The
  graph is genuinely multi-channel in its design, but the public data offers very few cross-channel connections.

### 5.6 A design change worth knowing

In the earlier version, the SMS "sender" was invented: the first phone number found in the text was treated as if it
were the sender. That was misleading, because the dataset has no senders. Now a phone number in the text is exactly
what it is: a `phone` node linked by `contains_phone`. The graph says "these messages mention the same number", which
is true and is precisely the kind of reuse scam campaigns show.

---

## 6. Stage 3: features

A model works on numbers, so every node must be described by a list of numbers called **features**.

### 6.1 Message features (33 numbers)

| Feature | Meaning |
|---|---|
| `length` | Number of characters |
| `digit_ratio` | Share of characters that are digits |
| `uppercase_ratio` | Share of characters that are capital letters |
| `url_count` | Number of links in the text |
| `phone_count` | Number of phone numbers in the text |
| `urgent_keyword_count` | How many of these words appear: urgent, verify, warning, refund, blocked, suspend, reset, failed, confirm |
| `channel_sms`, `channel_email`, `channel_url` | Which channel the message is from (one is 1, the others 0) |
| `text_000` … `text_023` | A 24-number summary of the words in the text |

**The 24-number text summary ("hashed bag of words").** Every word is assigned, by a fixed formula, to one of 24
slots, and the slot is increased or decreased by one. The result records roughly which words occur, without storing a
dictionary. *Why this?* It needs no download and no training, and it is the same for every channel. *Its weakness:* it
does not understand meaning. "Free prize" and "complimentary reward" look unrelated to it. A stronger alternative,
MiniLM sentence embeddings, is supported by the code but is **not used in the reported results** (see Part 19).

These are **content features**: they describe the message alone. The two baseline models see only these.

### 6.2 Features of the other nodes

| Node type | Features |
|---|---|
| sender, recipient, phone | number of edges going out, number of edges coming in, length of the value |
| url | the same three, plus number of slashes, number of digits, number of `? = &` characters, whether it starts with `https://` |
| domain | the same three, plus number of dots, digits and hyphens |

These are **structural features**: they describe how connected a node is. For example, a sender node with 50 incoming
edges has sent 50 emails in the data.

### 6.3 Scaling

Features have very different sizes (length can be 2,000 while a ratio is below 1). Before use, every feature is
rescaled to have average 0 and spread 1. Counts with a few huge values are log-transformed first. The averages used
for message features come **from the training messages only**, so nothing about the test data leaks in.

*This was missing in the earlier version* for the GNN, and adding it is one reason the GNN improved.

---

## 7. Stage 4: the models

### 7.1 Why there are baselines

A result like "the GNN gets 97%" means nothing on its own. It only means something next to simpler models on the same
data. A **baseline** is that simpler model.

| Model | Sees | Why it was chosen |
|---|---|---|
| Logistic regression | Content features of one message | The simplest standard classifier: a weighted sum of the features. If this already did well, the task would be easy. |
| Gradient-boosted trees | Content features of one message | One of the strongest standard methods for tables of numbers. A fair, hard-to-beat competitor. |
| Heterogeneous GraphSAGE (GNN) | Content features **and the graph** | The model under study. |

The earlier version had seven models. Four were removed (MLP, GCN, plain GraphSAGE, HGT) because they added nothing to
the argument, and two of them were unstable. Three models tell the story: simple baseline, strong baseline, your model.

### 7.2 What a GNN does, in plain words

A normal classifier takes the numbers of one message and outputs an answer. A GNN adds one idea: **before deciding,
each node collects information from its neighbours**. This is called *message passing*. After one round a node knows
about its direct neighbours; after two rounds it also knows about their neighbours.

**GraphSAGE** is one well-known way to do this: a node's new description is computed from its own description plus
the average of its neighbours' descriptions. *Heterogeneous* means there is a separate set of learned weights for each
edge type, so the model can learn, for instance, that "same sender" matters differently from "same website".

### 7.3 The architecture in this project (`ScamGNN` in `models.py`)

1. **Input layers.** Each node type has its own layer that turns its features into a list of 64 numbers.
2. **Context step.** Every sender collects information from its mail domain; every link collects information from its
   website.
3. **Two message-passing rounds.** Every message collects information from its sender, recipient, links, phone numbers
   and the messages it is directly connected to. The result is added to what the message already had and normalised.
4. **Classifier.** The message's original 64 numbers and its final 64 numbers are joined and passed through a small
   network that outputs two numbers: a score for "legitimate" and a score for "scam".

The model has 144,642 learned weights. It is *message-centric*: only message nodes are classified; all other nodes
exist to supply context.

*Why this design and not a deeper or fancier one?* Two rounds reach neighbours of neighbours, which is as far as the
relationships in this data meaningfully go. More layers would add weights without adding information, and the model
would be harder to explain.

**An important fact about what the GNN sees.** The labels of neighbouring messages are **never an input**. The model
does not look up "my neighbours are known scams". It only sees neighbours' *features* and the *structure*. Labels are
used solely to compute the training error.

---

## 8. Stage 5: training, validation, testing

### 8.1 Training the GNN

- The whole graph is given to the model, but the **error is measured only on training messages**.
- The model adjusts its weights to reduce that error (optimiser: Adam, learning rate 0.002).
- **Balanced weights:** each combination of channel and label counts equally in the error. Without this, the largest
  group (legitimate URLs) would dominate and SMS scams (385 training examples) would be learned badly.
- This repeats for up to 100 passes (called *epochs*).

### 8.2 Validation

After every epoch the model is scored on the **validation messages**. The best version so far is remembered. If there
is no improvement for 15 epochs in a row, training stops (*early stopping*). The demo model was best at epoch 46 and
stopped at epoch 61.

Validation data is also used to choose the **decision threshold**: the risk score above which a message is called a
scam. It is set to the value that gives the best F1-score on validation data. For the demo model it is 0.619.

*Why not just use 0.5?* Because the best cut-off depends on the model, and choosing it on validation data is fair to
every model. Choosing it on test data would be cheating.

### 8.3 Testing

The **test messages** are used once, at the end, to report results. They are never used to adjust weights, to pick the
best epoch or to pick the threshold.

### 8.4 One thing to state precisely: "transductive"

During training, the test messages *are present in the graph* (as nodes with features), but their labels are hidden.
This is the standard setting for GNNs and is called **transductive learning**. It is honest as long as you say it: the
model has seen the test messages' text and connections, but never their answers.

### 8.5 Seeds

Training starts from random weights, so results vary slightly from run to run. Every model is therefore trained three
times with different random seeds (7, 11, 42), and the average and spread are reported. The data split is the same in
all three runs.

### 8.6 The ablation

To find out whether the **graph** is what helps, the same GNN is trained in three variants:

| Variant | What it may use |
|---|---|
| Full | All edges |
| No message-to-message edges | Everything except `shares_sender`, `shares_recipient`, `shares_infrastructure` |
| No edges at all | Only the message's own content features |

If the full model beats the no-edges model, the graph is contributing. This is the most direct answer to the research
question.

---

## 9. Stage 6: evaluation

### 9.1 The metrics and what they mean

Start from four counts on the test data:

| | Model says SCAM | Model says LEGITIMATE |
|---|---|---|
| **Really a scam** | caught (true positive) | missed (false negative) |
| **Really legitimate** | false alarm (false positive) | correctly passed (true negative) |

That table is the **confusion matrix**. Every other threshold-based metric comes from it:

| Metric | Plain meaning | Formula |
|---|---|---|
| Accuracy | Share of all answers that were right | (caught + passed) ÷ all |
| Precision | When the model says scam, how often is it right? | caught ÷ (caught + false alarms) |
| Recall | Of all real scams, how many were caught? | caught ÷ (caught + missed) |
| F1-score | One number balancing precision and recall | 2 × P × R ÷ (P + R) |

Two metrics do not depend on a threshold; they judge the risk scores themselves:

| Metric | Plain meaning |
|---|---|
| ROC-AUC | The chance that a randomly chosen scam gets a higher risk score than a randomly chosen legitimate message. 0.5 is guessing, 1.0 is perfect. |
| PR-AUC | Precision and recall summarised over every possible threshold. It is the stricter one when scams are the minority. |

*Why not rely on accuracy?* Only 12% of SMS are scams. A model that always says "legitimate" would be 88% accurate and
catch nothing. Precision, recall and F1 expose that.

### 9.2 Results: model comparison (test data, average of 3 runs)

**F1-score**

| Model | SMS | Email | Web links | All channels |
|---|---|---|---|---|
| Logistic regression | 0.667 | 0.594 | 0.884 | 0.799 |
| Gradient-boosted trees | **0.911** | 0.816 | 0.968 | 0.935 |
| Heterogeneous GraphSAGE | 0.870 | **0.944** | **0.982** | **0.967** |

**PR-AUC**

| Model | SMS | Email | Web links | All channels |
|---|---|---|---|---|
| Logistic regression | 0.663 | 0.629 | 0.945 | 0.852 |
| Gradient-boosted trees | **0.964** | 0.898 | 0.994 | 0.983 |
| Heterogeneous GraphSAGE | 0.925 | **0.982** | **0.997** | **0.992** |

The spread between runs is small (at most 0.009 in F1), so the differences above are not run-to-run noise.

### 9.3 Results: the ablation (F1-score)

| GNN variant | SMS | Email | Web links | All channels |
|---|---|---|---|---|
| Full graph | 0.870 | **0.944** | 0.982 | **0.967** |
| No message-to-message edges | 0.874 | 0.921 | 0.983 | 0.963 |
| No edges at all | 0.870 | 0.809 | 0.958 | 0.922 |

### 9.4 What the results mean

1. **Overall, the GNN is the best model** (F1 0.967 against 0.935 for the strongest baseline).
2. **The graph is the reason, on email.** With edges removed the same GNN falls from 0.944 to 0.809, which is the same
   level as the best non-graph baseline (0.816). Email is the channel with real sender and recipient data.
3. **The graph helps a little on web links** (0.982 against 0.958 without edges). The gain comes from the link's
   website node, not from message-to-message edges.
4. **The graph does not help on SMS** (0.870 with or without edges), and gradient-boosted trees are better there
   (0.911). The SMS dataset has no sender information, so there is almost no graph to use: only 3% of SMS are linked to
   another message.
5. **The honest conclusion:** a graph-based model helps in proportion to how much real relationship information the
   data contains. That is a clear, defensible finding, and it answers the research question.

### 9.5 The demo model (seed 7) on the 5,190 test records

| | Accuracy | Precision | Recall | F1 | Caught | Missed | False alarms | Correctly passed |
|---|---|---|---|---|---|---|---|---|
| SMS | 96.5% | 86.0% | 86.0% | 0.860 | 111 | 18 | 18 | 886 |
| Email | 96.0% | 89.7% | 97.6% | 0.935 | 331 | 8 | 38 | 780 |
| Web links | 98.5% | 98.4% | 98.1% | 0.982 | 1,259 | 25 | 20 | 1,696 |
| **All** | 97.6% | 95.7% | 97.1% | 0.964 | 1,701 | 51 | 76 | 3,362 |

---

## 10. Stage 7: prediction (the detection workflow)

`predict.py` scores one new communication. The steps:

1. **Input.** Channel, text, and for email optionally the sender and recipient addresses.
2. **Checks.** Empty input, input with no letters or digits, text over 2,000 characters and things that are not a web
   link are rejected with a clear message.
3. **Event row.** The input is cleaned exactly like the training data.
4. **Connect to the stored graph.** A new message node is created. Its links, phone numbers, sender and recipient are
   looked up: if they already exist in the graph, the new message is attached to them (and to the stored messages that
   share them); if not, new nodes are created.
5. **Features.** The 33 content numbers are computed for the new message. The edge counts of any existing node that
   gained an edge are updated.
6. **Model.** The trained GNN runs once over the graph.
7. **Output.** The risk score of the new node (0 to 1). If it is at or above the threshold (0.619), the verdict is SCAM.

Nothing is saved: the next message starts from the same stored graph.

**Speed.** About 0.4 seconds per message on the development laptop. The system adds the new message to the stored
graph instead of rebuilding everything. A test (`tests/test_prediction.py`) confirms that this gives the same score as
rebuilding the whole graph from scratch.

**What "real-time" means here, precisely.** The system gives a verdict for one message in under a second, on demand,
and the verdict banner is the alert. What is **not** built: a live feed of incoming messages, automatic notifications,
or handling of many messages per second. Describe it as *on-demand, near-real-time scoring of a single message*.
A live deployment is future work.

---

## 11. Stage 8: explanation

**Method: remove one factor and score again** (also called perturbation or occlusion analysis).

For a scored message, `explain.py` looks for these factors: a phone number in the text, a link in the text, urgent
words, the sender address, the recipient address, and connections to other messages. For each one that is present, it
removes just that factor, scores the message again, and reports the new risk score.

Example from the demo (the SMS "Hi mum, this is my new number 08000839402…"):

| What was removed | Risk score without it |
|---|---|
| (nothing) | 99.9% |
| Phone number in the text | 0.2% |
| Connections to other messages | 99.8% |

Reading: the phone number is what drives this score.

*Why this method?* It uses the real model, it is easy to explain in one sentence, and it is a recognised technique.
*What it does not do:* it does not reveal the model's internal reasoning, and the effects of several factors need not
add up. A dedicated tool such as GNNExplainer is future work.

---

## 12. What the model actually relies on

You should know this better than your examiners. These behaviours were measured, not assumed.

- **SMS:** the content of the text dominates. Removing the phone number or urgent words changes the score sharply;
  removing connections changes almost nothing.
- **Email:** relationships dominate. The same harmless email text scores 0.2% when sent from an address that has sent
  55 other emails in the data, and about 20% from a completely unknown address. Regular correspondents look legitimate.
- **A consequence of the model not seeing labels:** an email from a sender with 16 previous messages, *all spam*, with
  harmless wording also scores low (0.5%). The model has learned "senders who appear many times are usually
  legitimate", which is true on average in this data, but it cannot tell a repeat spammer by the count alone.
- **Unknown email senders lean towards scam.** In the data, 62% of unlinked emails are spam.

---

## 13. The demonstration (UI)

One page, top to bottom (`app/app.py`, built with Streamlit):

1. **Analyse a message.** Choose a channel (SMS, Email, Web link). Click an example button or type your own. For email
   there are optional sender and recipient boxes. The button *Random real record from the test data* shows a real
   stored record with its true answer.
2. **Result.** A red SCAM or green LEGITIMATE banner with the risk score; the "remove one thing" table; the graph
   picture around the message; and a closed *Technical details* section.
3. **How good is the system?** The model comparison table, the ablation table, the four confusion-matrix counts, and
   closed sections for more metrics, live tests, the data and graph, and limitations.

**How the page and the model interact.** There is no separate server. The page is a Python script; it creates one
`Predictor` object when it starts (which loads `results/model.pt` and `results/prepared.pkl`) and calls
`predictor.predict(...)` and `explain(...)` when you press Analyze. The results tables are read from
`results/results.json`. The page never contains a typed-in number.

**The prepared test inputs** (`app/test_cases.json`): 17 inputs with the expected answer written before running the
model. Current score: **13 of 17 correct**. The four wrong ones are kept on purpose:

| Input | Expected | Model | Why it is instructive |
|---|---|---|---|
| SMS "Congratulations! You have been selected for a cash reward…" | SCAM | LEGITIMATE (0.4%) | Scam wording with no number, link or urgent keyword is missed |
| SMS "Hi mum, this is my new number 08000839402…" | LEGITIMATE | SCAM (99.9%) | A harmless message containing a number used by spam is flagged |
| Email "Seminar room change…" from an unknown sender | LEGITIMATE | SCAM (97%) | Unknown email senders lean towards scam |
| Email advance-fee fraud | SCAM | LEGITIMATE (57%) | Just under the 62% decision line |

---

## 14. How to run everything

| Task | Command | Time |
|---|---|---|
| Start the demo | double-click `run_demo.bat` | page ready in about 15 seconds |
| Re-train and re-evaluate | `python -W ignore -m scam_gnn.train` (with `PYTHONPATH=src`) | about 14 minutes |
| Run the tests | `python -W ignore -m unittest discover -s tests -v` | about 1 minute |

Requirements: Python 3.11+, and the packages in `requirements.txt` (numpy, pandas, scikit-learn, torch,
torch-geometric, streamlit, matplotlib). No internet connection is needed to train or to demonstrate.

## 15. How the system is tested

15 automated tests:

- **Stage tests** (`tests/test_pipeline_stages.py`, 9 tests): links, phone numbers and addresses are found correctly;
  duplicates are removed; the split has the right proportions and is reproducible; the graph has the expected nodes and
  edges, including a cross-channel link; the GNN runs in all three ablation variants; metrics give the right values on
  a hand-worked example.
- **System tests** (`tests/test_prediction.py`, 6 tests): the reloaded model reproduces the scores saved during
  training; fast scoring equals a full graph rebuild for every channel; predicting does not alter the stored graph;
  the explanation finds the factors; invalid input is rejected; a stored record returns its saved score.

---

## 16. Known limitations

1. **Old, public datasets.** SMS from about 2012, email from about 2002. Modern scam styles are under-represented.
2. **Spam is not exactly scam.** Two datasets are labelled spam versus legitimate.
3. **No sender data for SMS and web links,** so the graph adds nothing on SMS.
4. **Almost no cross-channel links** (8 edges). The multi-channel graph is real in design; the data rarely connects
   channels.
5. **The email result depends on the corpus.** Legitimate SpamAssassin email is largely mailing-list traffic between
   regular correspondents, which makes sender and recipient patterns unusually informative. Expect a smaller benefit on
   ordinary personal mailboxes.
6. **The model does not use neighbours' labels,** so a repeat spammer with harmless wording can be missed.
7. **Random split.** It does not test how well the model handles scams that appear after training.
8. **Simple text features.** Wording without numbers, links or keywords is often missed.
9. **Transductive evaluation.** Test messages are in the graph during training, without labels.
10. **Not a live system.** One message at a time, on one laptop.
11. **The explanation is a perturbation test,** not a view inside the model.
12. **No calls** (Part 18).

## 17. How the project evolved

### 17.1 Where it started

The original code was a research scaffold. An inspection found:

- No trained model was saved anywhere, and no code could score a new message.
- Seven models, two of them unstable.
- A second, separate web dashboard of about 4,100 lines, larger than all the machine-learning code, which could not
  analyse a new message.
- On the original data (SMS and URL only), **every GNN lost to the non-graph baselines**.
- 97% of messages had no link to any other message.
- 91 test SMS were word-for-word copies of training SMS.
- The "time-based" split used invented timestamps; the SMS "sender" was invented from the text.
- URL results were near-perfect because of ready-made dataset columns.
- Unrelated files in the folder, and an author field naming an AI tool.

### 17.2 What was changed and why

| Change | Problem it solved |
|---|---|
| Added the email channel (SpamAssassin) | The data had no real relationships, so a graph model had nothing to use |
| Removed duplicates before splitting | Test data leaking from training data |
| Honest stratified random split | A "time-based" split built on invented timestamps |
| Phone numbers became `phone` nodes instead of fake senders | A misleading graph |
| Dropped PhiUSIIL's ready-made columns | Trivially perfect URL scores; links could not be typed live |
| One uniform feature set for all channels | Every channel can now be scored live |
| Feature scaling for the GNN | The GNN was trained on badly scaled inputs |
| Longer training with early stopping (up to 100 epochs) | The earlier 40-epoch limit stopped training too early |
| Threshold chosen on validation data | A fixed 0.5 threshold that suited no model in particular |
| Graph ablation added | The research question had never been tested directly |
| Three models instead of seven | A diluted comparison |
| The pipeline now saves its model, and has `predict.py` and `explain.py` | No saved model; no way to score or explain a new message |
| One-page demo; old dashboard archived | Two user interfaces; a confusing demo |
| Baselines no longer receive graph-degree features | The "non-graph" baselines were quietly using graph information |

### 17.3 Technical challenges met along the way

- **Start-up time.** Rebuilding the graph every time the demo opened took about 40 seconds and made the page
  unresponsive. Fixed by saving the prepared graph.
- **Scoring speed.** Rebuilding the whole graph for each new message took about 10 seconds. Fixed by attaching the new
  message to the stored graph; now about 0.4 seconds, verified equal to a full rebuild.
- **A very slow internet connection** during the upgrade. The originally chosen email corpus (65 MB) could not be
  downloaded, so the 14 MB SpamAssassin corpus from the same collection was used.
- **Before and after are not directly comparable.** The earlier numbers were on different data, a different split and
  different features. Do not present them as a like-for-like improvement.

---

## 18. The call-data decision

### 18.1 The original intention

The project title speaks of *multi-channel communications*, and phone calls are one of the main channels scammers use.
The broader concept is a system in which SMS, email, web links **and calls** all become nodes in one graph, so that a
scam campaign using the same phone number to send texts and to place calls is recognised as one campaign. Calls fit the
concept naturally: the graph already has a `phone` node type.

### 18.2 Why calls are not implemented in the current graph environment

This was a design decision based on four concrete problems, not simply "there was no dataset".

**1. No suitable public data exists in the form the graph needs.** The graph is built from *identifiers that messages
share*: sender, recipient, phone number, link. For calls, that means call records: who called whom, when, labelled
scam or not. Phone companies do not publish those. What is publicly available is:

| What exists | Why it does not fit |
|---|---|
| AI-generated call transcripts (labelled scam / not scam) | Not real calls; no caller or receiver identifiers |
| Transcripts of real scam calls taken from online videos | Scam only: there are no legitimate calls to compare with, so nothing can be trained or evaluated |
| A set of 60 labelled transcripts | Far too small to train or test on |
| Public complaint lists of caller numbers | No call content, no legitimate class, unverified reports |

**2. Transcripts would be isolated nodes.** A transcript has no caller number, so a call node would have no edges. A
GNN applied to a node with no edges is just an ordinary text classifier. Adding calls this way would add a channel to
the title but nothing to the research question, which is about relationships. The SMS results already show what
happens when a channel has no relationship data: the graph gives no benefit.

**3. A call is audio, not text.** To use a real call the system would need speech recognition first. That is a
separate system with its own errors, and the project would then be evaluating speech recognition as much as scam
detection. The current features (short-text counts and a 24-number word summary) are also not designed for long
spoken conversations.

**4. It would weaken the evaluation.** AI-generated transcripts are typically easy to tell apart, so they would
produce another near-perfect score that proves little. The earlier version of this project had exactly that problem
with the URL data, and removing it was one of the improvements.

### 18.3 Why SMS, email and web links are used

| Channel | Why it is in |
|---|---|
| SMS | A real, labelled public dataset; phone numbers and links inside messages give some graph structure |
| Email | A real, labelled public dataset **with senders and recipients**: the channel where a graph model can show its value |
| Web links | A large real, labelled public dataset; links are the element that most naturally connects channels, because SMS and emails contain links |

Together they cover three different kinds of communication (short text, long text with addressing, and a bare link),
all with real labelled data, in one graph with one model. That is a meaningful multi-channel implementation, and the
three channels differ in how much relationship data they have, which is exactly what makes the comparison informative.

### 18.4 Why this does not invalidate the project

Keep two things separate when you speak:

- **The concept:** a graph-based detector for communications across channels. The design is channel-independent: a
  channel is added by converting its records to the common event format.
- **The current version:** three channels are implemented and evaluated.

The research objective is to test whether graph-based modelling improves scam detection across more than one channel.
That is tested and answered with three channels. Calls would be a fourth instance of the same idea, not a different
idea. Scoping a project to the data that can support a valid evaluation is normal research practice; including a
channel that could not be validly evaluated would have been the weaker choice.

### 18.5 Future integration of calls (future work, not current functionality)

If call records with caller and receiver numbers and reliable labels became available (for example through a
partnership with a telecom operator), calls could be added as follows:

1. Each call becomes a `message` node with channel `call`.
2. The caller's number becomes a `phone` node, which **already exists** in the graph. A call from a number that also
   appears in scam SMS would be connected to those SMS automatically. This is the cross-channel link the concept was
   designed for.
3. Call features would be metadata (duration, time of day, how many different people the caller rang) and, where a
   transcript is available, text features from it.
4. The model and training code would not need to change.

---

## 19. The recommendation that was implemented, and other future work

**The recommendation:** make the project smaller, correct and aligned with its research question: remove what is
unused, fix the data handling, add a channel with real relationships (email), compare three models, run the ablation,
save the model, add prediction and explanation stages, and present it as one clear pipeline with a one-page demo.

**Why it was chosen:** the earlier version could not support its own central claim, and its size hid that. A smaller
project in which every part can be explained and every number reproduced is stronger in a defence.

**Not completed: MiniLM text embeddings.** The plan included replacing the 24-number word summary with MiniLM sentence
embeddings, which capture meaning. The code supports it (`"text_embedding": "minilm"` in `config.json`), but the
required download did not complete during the upgrade because of the internet connection, so **all reported results
use the hashed word summary**. This is future work.

**Other future work**

- Call data, as described in 18.5.
- Data with real SMS senders, and data that genuinely links channels.
- Feeding known labels of neighbouring messages into the model, so that repeat spammers are recognised.
- A time-based evaluation on data that has real timestamps.
- A live deployment: a message feed, alerts, and a graph database.
- A dedicated GNN explanation method.
- Newer datasets.

---

## 20. Defence preparation

The answers are written so that you understand them. Say them in your own words.

### The problem

**What problem does your project solve?**
Scam detection that uses relationships. Scammers reuse phone numbers, websites and sender addresses. Looking at one
message at a time ignores that. I represent messages and what they share as a graph and test whether a graph neural
network detects scams better than models that see each message alone.

**What is your research question, and what is the answer?**
Does a graph-based model beat non-graph models on multi-channel scam detection? Answer: yes overall (F1 0.967 against
0.935), and the benefit is proportional to how much relationship data a channel has. Large on email, small on web
links, none on SMS.

### The datasets

**Which datasets did you use and why?**
UCI SMS Spam Collection, the SpamAssassin email corpus and the PhiUSIIL URL dataset. All are public, real and
labelled. Email was added specifically because it has senders and recipients.

**Are they scam datasets?**
Not exactly. SMS and email are labelled spam versus legitimate; URLs are labelled phishing versus legitimate. I treat
all three as "unwanted or malicious" versus "legitimate". That is a limitation I state.

**How did you prevent data leakage?**
Duplicates are removed before splitting (444 records). Feature scaling uses training data only. The threshold and the
best epoch are chosen on validation data. Test data is used only for the final numbers.

**Why did you sample the URL dataset?**
It has 235,795 rows against about 5,000 in each other channel. I took a random 15,000 with the original phishing rate
so that one channel does not dominate and training fits on a laptop.

**Why did you drop the extra PhiUSIIL columns?**
They made the task trivial and cannot be computed for a new link. Using only the URL keeps the evaluation meaningful
and lets links be scored live.

**Why a random split and not a time-based one?**
Two of the three datasets have no timestamps. Inventing them would be misleading.

### The multi-channel design

**What makes it multi-channel?**
All three channels are converted to one event format and placed in one graph with shared node and edge types. One
model classifies all of them.

**Are the channels actually connected to each other?**
Barely: 8 edges, between emails and URL records on the same website. The datasets were collected independently. The
design supports cross-channel links; the public data provides very few.

### The graph

**What are your nodes and edges?**
Six node types: message, sender, recipient, url, domain, phone. Nine edge types: sent_by, sent_to, contains_url,
contains_phone, resolves_to_domain, uses_domain, and three message-to-message edges for a shared sender, recipient, or
phone number or website. In total 66,888 nodes and 324,229 edges.

**Why are there message-to-message edges if messages are already connected through shared nodes?**
They let information pass directly between two related messages in one step, which is what the model's message layers
use.

**Why the limit of 128 on group size?**
A thing shared by hundreds of messages, such as a mailing-list address, says little about any one of them, and
connecting every pair would create millions of edges.

**Why heterogeneous?**
The nodes are different kinds of things with different features, and "same sender" should not be treated like "same
website". A heterogeneous model learns separate weights for each.

### Why GNNs, and why not traditional ML alone

**Why a GNN?**
Because the hypothesis is about relationships, and a GNN is the standard type of model that learns from both features
and connections.

**Why not just use gradient-boosted trees?**
I did, as a baseline, and on SMS they are better (0.911 against 0.870), because SMS has no relationship data. On email
the GNN is much better (0.944 against 0.816). So traditional ML is enough when there are no relationships and falls
short when there are.

**How do you know the improvement comes from the graph and not just from a bigger model?**
The ablation. The same GNN with its edges removed drops on email from 0.944 to 0.809, the level of the non-graph
baseline.

### Model architecture

**Describe your model.**
A heterogeneous GraphSAGE. Each node type has an input layer producing 64 numbers. Senders and links first collect
information from their domains. Then, twice, each message collects information from its sender, recipient, links, phone
numbers and connected messages. A final small network outputs scam or legitimate. 144,642 weights.

**Why two layers?**
Two rounds reach neighbours of neighbours, which covers the relationships in this data. More layers add complexity
without adding information.

**Does the model see the labels of neighbouring messages?**
No. Only their features and the structure. That is also why a repeat spammer with harmless wording can be missed.

### Training

**How was it trained?**
Adam optimiser, learning rate 0.002, up to 100 epochs, early stopping after 15 epochs without improvement on
validation data, error weighted so that every channel and class counts equally, three random seeds.

**What is the validation set for?**
Choosing the best epoch and the decision threshold, so the test set stays untouched.

**What does "transductive" mean in your project?**
Test messages are in the graph during training without their labels. The model has seen their content and
connections but not their answers. This is the standard GNN setting.

### Testing and evaluation

**Which metric matters most and why?**
F1-score and PR-AUC, because scams are the minority and accuracy can look good while catching nothing.

**What is your threshold and how was it chosen?**
0.619 for the demo model: the value with the best F1 on validation data.

**How reliable are the numbers?**
Each model was trained three times; the spread in F1 is at most 0.009. Everything is reproducible with one command.

**How did you test the software?**
15 automated tests, including one proving that the fast scoring used in the demo equals a full rebuild of the graph.

### Results

**Summarise your results.**
Overall F1: GNN 0.967, gradient-boosted trees 0.935, logistic regression 0.799. Email: GNN 0.944 against 0.816. Web
links: 0.982 against 0.968. SMS: trees 0.911 against GNN 0.870.

**Why does your model lose on SMS?**
The SMS dataset has no senders, so only 3% of SMS are linked to another message. With nothing to pass along the graph,
the GNN is an ordinary classifier there, and boosted trees are very strong on small tables.

**Is the email result too good to be true?**
It is real but depends on the corpus. Legitimate SpamAssassin email is mostly mailing-list traffic between regular
correspondents, so who-writes-to-whom is very informative. I would expect a smaller gain on ordinary mailboxes.

### Real-time detection

**Is it real-time?**
It scores one new message in about 0.4 seconds on a laptop. I describe it as on-demand, near-real-time scoring. A live
feed, alerts and high throughput are not built.

**How can a new message be scored if the model needs a graph?**
The new message is attached to the stored graph through any sender, recipient, link or phone number it shares, then
the trained model is run. Nothing is saved afterwards.

### The UI

**What does the demo show?**
A message goes in; the verdict and risk score come out; a table shows how the score changes when each factor is
removed; a picture shows the graph around the message; and the evaluation tables are at the bottom.

**Are the numbers in the demo typed in?**
No. They are read from `results/results.json`, written by the training command.

**Your demo gets some examples wrong. Why show them?**
Because they are what the model does. 13 of 17 prepared inputs are correct, and each wrong one illustrates a specific
limitation.

### Limitations

**What are the main limitations?**
Old datasets; spam rather than strictly scam labels; no sender data for SMS and links; almost no cross-channel links;
the model does not use neighbours' labels; a random rather than time-based split; simple text features; not a live
system.

**Can a scammer evade it?**
Yes. Scam wording with no number, link or urgent keyword is often missed, as one of my test inputs shows.

### The decision not to implement calls

**Your title says multi-channel. Why are there no calls?**
The graph is built from identifiers that messages share. Public call data is either AI-generated transcripts, scam-only
transcripts or tiny sets. None has caller identifiers or a usable legitimate class. A transcript would be a node with
no edges, so the GNN would act as a plain text classifier and the result would say nothing about my research question.
Real calls are also audio and would need speech recognition first.

**Does that mean the project is incomplete?**
No. The objective is to test graph-based detection across multiple channels, and it is tested on three. Calls would be
a fourth instance of the same design.

**Why are SMS, email and web links sufficient?**
They are three different kinds of communication with real labelled data, and they differ in how much relationship
information they carry, which is what lets me show when a graph helps.

**How would you add calls?**
With call records containing caller numbers. The caller number becomes a `phone` node, which already exists, so a call
would automatically connect to SMS mentioning the same number. The model would not change.

### Future work

**What would you do next?**
Obtain data with real SMS senders and real cross-channel links; add calls from call records; use sentence embeddings
for text; let the model use known labels of neighbours; evaluate on a time-based split; build a live pipeline.

### About the work itself

**Which parts did you build, and how was AI assistance used?**
Answer this truthfully and in your own words. You should be able to explain every part of this handbook without
reading it, and to say which decisions were yours.
