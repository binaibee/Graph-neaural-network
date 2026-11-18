# Presentation guide

Everything here describes what the code in this folder actually does. Every number comes from running the project on
this laptop (demo model: heterogeneous GraphSAGE, seed 7).

## Part 1 — How the system works, in plain language

### 1. What happens when I enter a message
The text you type is turned into one row of data, added to a large network of earlier messages, and the trained model
calculates a number between 0 and 1 for it. That number is the risk score. If it is 0.5 or more, the demo says SCAM.

### 2. How the message becomes data
- Extra spaces are removed and the channel is set to `sms`.
- The system looks for a phone number in the text, and if there is none, a link. The first one found becomes the
  "sender". If there is neither, the sender is a placeholder called "unknown". This is the same rule the project
  applies to the UCI SMS dataset, because that dataset has no real sender column.
- The message is then described by **84 numbers** (its features):
  - 5 simple counts: length, share of digits, share of capital letters, number of links, number of urgent keywords
    (`urgent`, `verify`, `warning`, `refund`, `blocked`, `suspend`, `reset`, `failed`, `confirm`);
  - 2 channel flags (is it an SMS, is it a URL);
  - 24 numbers that summarise which words appear (a "hashed bag of words": every word is assigned to one of 24 slots);
  - 50 page statistics that exist only for URL records (they are 0 for an SMS);
  - 3 counts of how many connections the message has in the graph.

### 3. How the graph is constructed
A graph is a set of dots (nodes) joined by lines (edges). For every message the code creates a message node and
connects it to the things it mentions. If two messages share the same sender, recipient or web domain, they are also
connected directly to each other. The full graph has **91,816 nodes and 79,835 edges**, built from **20,574 messages**
(5,574 SMS and 15,000 URLs).

### 4. What the nodes represent
| Node | Meaning |
|---|---|
| `message` | One SMS or one URL record. This is the thing being classified. |
| `sender` | Who sent it (for SMS: the phone number or link found in the text) |
| `recipient` | Who received it (a placeholder for SMS; the domain for URL records) |
| `phone` | A phone number |
| `email_address` | An email address |
| `url` | A full link |
| `domain` | The website part of a link, for example `www.urawinner.com` |

### 5. What the edges represent
| Edge | Meaning |
|---|---|
| `sent_by`, `sent_to` | message → its sender / recipient |
| `uses_phone`, `uses_email` | sender or recipient → the phone number / email address it uses |
| `contains_url` | message → a link inside it |
| `resolves_to_domain` | link → its website domain |
| `shares_sender`, `shares_recipient`, `shares_infrastructure` | message ↔ another message with the same sender, recipient or domain |

### 6. What information enters the GNN
Two things: the feature numbers of every node, and the list of edges. **The labels (scam / legitimate) are never an
input.** They are only used during training to tell the model whether its answer was right.

### 7. What the GNN actually does
A normal classifier looks at one message alone. A Graph Neural Network also lets each node collect information from
its neighbours. In this model:
1. Each node's features are converted into a list of 64 numbers.
2. Senders collect information from their phone numbers, and links collect information from their domains.
3. Twice in a row, each message collects information from its sender, recipient, links and directly linked messages.
   After two rounds a message "knows" about its neighbours and its neighbours' neighbours.
4. A small final network turns the result into two numbers: one for "legitimate" and one for "scam".

The model is a heterogeneous GraphSAGE. "Heterogeneous" means it uses separate learned weights for each edge type.

### 8. What the output means
The output is the model's estimated probability that the message is a scam. It is a statement about patterns in the
training data, not a proven fact about the message.

### 9. How the risk score is produced
The two output numbers are passed through a softmax, which rescales them so they add up to 1. The "scam" share is the
risk score. The demo's "Confidence" (High / Medium / Low) only describes how far the score is from 0.5. It is a display
choice, not something the model outputs.

### 10. How the model was trained
- All messages are ordered and split: first 60% **train** (12,344), next 20% **validation** (4,115), last 20% **test** (4,115).
- The model sees the whole graph, but its error is measured only on training messages. It adjusts its weights to
  reduce that error (Adam optimiser, learning rate 0.002, at most 40 passes).
- Rare groups are given more weight so that the model does not ignore the minority class.
- After each pass it is checked on the validation messages. The best version is kept, and training stops early if
  there is no improvement for 8 passes.

### 11. How the test set differs from the training set
Test messages are the last 20% of the data. Their labels are never used to adjust the model or to choose the best
version, so test results show how the model performs on messages it was not taught the answer for.

One point to state precisely: test messages are present in the graph during training (without their labels). This is
called **transductive** learning and is standard for GNNs, but it is not the same as never having seen the message.

### 12. What each evaluation metric means
Numbers below are for the demo model on the test split.

| Metric | Plain meaning | Overall | SMS only |
|---|---|---|---|
| Precision | Of the messages flagged as scam, how many really were | 0.930 | 0.574 |
| Recall | Of the real scams, how many were caught | 0.975 | 0.800 |
| F1-score | A single number balancing precision and recall | 0.952 | 0.669 |
| Accuracy | Share of all predictions that were right | 0.966 | 0.897 |
| ROC-AUC | Chance that a random scam gets a higher score than a random legitimate message | 0.997 | 0.948 |
| PR-AUC | Summary of precision and recall across all thresholds; the project's main metric because scams are the minority | 0.995 | 0.803 |
| Recall @ 90% precision | How many scams can be caught while keeping 9 in 10 alerts correct | 0.989 | 0.621 |

Confusion matrix (overall): 1,385 scams caught, 36 scams missed, 104 false alarms, 2,590 legitimate correctly passed.

### 13. Limitations
1. **Baselines beat the GNN.** Across 3 seeds the gradient-boosted trees reach PR-AUC 0.999 overall and 0.970 on SMS;
   the best GNN reaches 0.994 and 0.818.
2. **The URL dataset is too easy.** Its ready-made columns separate phishing from legitimate URLs almost completely
   (`URLSimilarityIndex` alone is right about 99.5% of the time; removing it changes nothing because other columns do
   the same job), so every model scores near 1.0 on URLs.
3. **SMS graph links come from the text.** There is no real sender data. Every sender shared by more than one SMS
   (101 of them) is 100% spam, and 99.9% of legitimate SMS have no phone number or link at all.
4. **The model leans on "contains a phone number or link".** Scam wording without one is missed (test cases T5, T6);
   a harmless message with one is flagged (T8).
5. **The graph links add little.** For T3 the score is 99.0% with links and 97.4% with the links removed.
6. **No cross-channel links exist in the data.** SMS and URL records share no entities, so "multi-channel" here means
   two channels in one graph, not connections between channels.
7. **Timestamps are synthetic**, so the "time-based" split is a split by dataset order.
8. **Text representation is basic**: a 24-slot hashed bag of words, because the MiniLM embedding model named in the
   config is not installed.
9. **Speed**: about 2 to 3 seconds per message on a laptop. This is on-demand scoring, not a production real-time system.
10. **No explainability method** is implemented.

## Part 2 — Demo script

| Step | What to do | What to say |
|---|---|---|
| 1 | Before the session, double-click `run_demo.bat` and wait until the four example buttons appear. | — |
| 2 | Show the title. | "Scam messages reuse the same phone numbers and links. This project tests whether modelling those relationships as a graph helps detect them. The model loaded here is a heterogeneous GraphSAGE trained on 20,574 real SMS and URL records." |
| 3 | Tab 1: click the example button **Scam with a phone number**. | "The message becomes a new node in the graph and the trained model scores it. This takes about 3 seconds." |
| 4 | Point to the red SCAM banner and the risk score. | "SCAM, 99%. The score is the model's probability for the scam class; 50% is the threshold." |
| 5 | Scroll to the graph. | "This is what makes it graph-based: the message, the phone number it contains, and 15 earlier messages using the same number." |
| 6 | Read the "Why?" lines. If asked for more, open **Technical details**. | "These are the actual inputs. With the links removed the model still gives 97%, so most of the signal comes from the message's own features. I'm showing that because it is what the model really does." |
| 7 | Click the example button **Scam with no number or link**. | "This is a clear scam to a human, but the model says legitimate at 27%. It has no phone number or link, and in the training data that is what almost all spam has. This is a limitation of the dataset and the simple text features." |
| 8 | Tab 2, Test B: show the table, press **Pick 12 different records** once. | "These are real test records with known labels, drawn at random, not hand-picked." |
| 9 | Tab 2, Test A: press **Run the 10 tests**. | "Ten messages written before running the model: 7 of 10 correct. The three errors are the two no-link scams and a harmless message containing a spam number." |
| 10 | Tab 3: show the table and the four counts; point to the SMS row. | "Overall PR-AUC 0.995. On SMS alone it is 0.80, which is the harder and more honest number." |
| 11 | Open **Comparison with the other models in the project**. | "I compared seven models over three seeds. On this data the non-graph baselines outperform the GNNs. My conclusion is that public SMS and URL datasets lack the relational data a GNN needs." |
| 12 | Open **Limitations**. | Read the main points and state what you would do next (below). |

**What you would do next:** obtain data with real sender and recipient information; add an email dataset so channels
actually connect; install the MiniLM text embeddings the config already names; remove `URLSimilarityIndex` and re-test.

## Part 3 — Likely questions and honest answers

**Is this really a GNN, or a text classifier with a graph drawn on top?**
It is a real GNN: a heterogeneous GraphSAGE in PyTorch Geometric, and the score shown is its output for the message
node. But on this data the graph contributes little, and the demo measures that (score with and without links).

**Why do your baselines beat your GNN?**
Because the datasets have almost no genuine relationships. UCI SMS has no sender information, and the URL features
already separate the classes almost perfectly. A GNN helps when relationships carry information that the features do
not; here they do not.

**Then what is your contribution?**
A working pipeline that fuses two public datasets into one heterogeneous graph, a fair comparison of seven models over
three seeds with time-ordered splits, and evidence about when graph structure does and does not help.

**Is it real-time?**
It scores one new message in about 2 to 3 seconds on a laptop CPU. I would call it on-demand inference. A production
real-time system would need incremental graph storage and batching, which I have not built.

**Is it multi-channel?**
It handles SMS and URLs in one graph and one model. The two datasets share no entities, so there are no edges between
channels. Email is supported by the code but no email dataset is included.

**Did the model see the test data?**
It saw the test messages' features and connections, never their labels. That is the standard transductive setup for
GNNs. Labels of the last 20% were used only for the final evaluation.

**Why PR-AUC and not accuracy?**
Only 13% of SMS are spam. A model that says "legitimate" every time would get 87% accuracy and catch nothing. PR-AUC
focuses on how well the scams are found.

**Why is SMS precision only 0.57?**
At the 0.5 threshold the model raises 86 false alarms against 116 correct detections on test SMS. The training weights
push it to catch more scams (recall 0.80) at the cost of more false alarms.

**Why are URL scores almost perfect?**
The PhiUSIIL dataset ships with engineered columns that separate the classes almost completely; even a single column
(`URLSimilarityIndex`) is right about 99.5% of the time. That is a property of the dataset, not of my model.

**Can it explain its decisions?**
No. The demo shows the inputs and the graph connections, and one measured comparison (with and without links). A proper
method such as GNNExplainer would be future work.

**How do you know the demo matches the research code?**
The demo calls the project's own functions for graph building and features, and a test checks that its score equals
the score from rebuilding the entire graph with the original pipeline. On start-up it also checks that the reloaded
model reproduces the scores saved during training.

**Could someone fool it?**
Yes. A scammer who writes without a phone number or link would likely pass, as test cases T5 and T6 show.

**How was the AI assistance used?**
Answer this truthfully in your own words. You should be able to say which parts you directed, and you should be able
to explain every section of this guide without reading it.
