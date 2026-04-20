# Scam Detection Analyst UI PRD

## 1. Product Summary

### Working name

Scam Intelligence Workbench

### Product type

An internal web application for fraud analysts, investigators, and ML researchers to:

- triage suspicious messages and URLs
- investigate linked entities and campaigns across channels
- review model outputs and explanations
- compare experiment runs and error patterns

### What the UI does in plain language

The UI turns raw scam detections into an investigation workspace. Instead of looking at one suspicious SMS or URL in isolation, the user can see:

- what was flagged
- why it was flagged
- what sender, recipient, URL, and domain it is connected to
- whether it belongs to a larger campaign
- how the model is performing by channel and over time

This is not a consumer-facing app. It is an internal operations and research console.

## 2. Users

### Primary user 1: Fraud analyst

Needs to review alerts quickly, decide whether something is a scam, and escalate real campaigns.

### Primary user 2: Investigator

Needs to follow connections across entities, channels, and infrastructure to understand campaign behavior.

### Primary user 3: ML researcher

Needs to compare model runs, inspect false positives and false negatives, and understand where graph context helps.

### Secondary user: Team lead

Needs high-level dashboards showing queue size, model quality, campaign trends, and analyst throughput.

## 3. Product Goals

### Goal 1

Help analysts move from single-alert review to relationship-aware investigation.

### Goal 2

Make cross-channel scam campaigns visible through graph, timeline, and entity views.

### Goal 3

Give researchers a UI to compare baselines vs GNN models without reading raw JSON artifacts.

### Goal 4

Keep evidence traceable. Every score should link back to messages, entities, edges, and model outputs.

## 4. Non-Goals For V1

- Real-time blocking or automated enforcement
- Public user reporting portal
- Voice or call-audio analysis
- Case-management workflow as deep as a full SOC platform
- Collaborative annotation beyond basic status, notes, and labels

## 5. Product Modes

The UI has two modes in one product.

### Mode A: Analyst Console

Focuses on alerts, investigations, campaigns, and entity risk.

### Mode B: Research Lab

Focuses on datasets, experiment runs, metrics, ablations, and error analysis.

## 6. Core Objects In The UI

### Alert

A scored suspicious artifact shown to the user for triage.

Fields:

- `alert_id`
- `message_id`
- `channel`
- `timestamp`
- `risk_score`
- `status`
- `model_version`
- `top_reasons`

### Message / Artifact

The unit being classified.

Fields:

- `message_id`
- `channel`
- `text_or_url`
- `label`
- `predicted_score`
- `sender`
- `recipient`
- `linked_urls`
- `linked_domains`
- `split`

### Entity

A sender, recipient, phone, email, URL, or domain.

Fields:

- `entity_id`
- `entity_type`
- `entity_value`
- `risk_mean`
- `risk_max`
- `linked_message_count`
- `first_seen`
- `last_seen`

### Campaign

A cluster of related suspicious artifacts connected by shared senders, recipients, or infrastructure.

Fields:

- `campaign_id`
- `size`
- `channels`
- `top_entities`
- `time_span`
- `risk_level`

### Experiment Run

A single training/evaluation result.

Fields:

- `run_id`
- `dataset_name`
- `model_name`
- `seed`
- `metrics`
- `channel_slices`
- `artifact_path`

## 7. Information Architecture

### Page 1: Dashboard

Purpose:

- give a fast overview of current scam activity, analyst workload, and model performance

Key content:

- total alerts
- high-risk alerts in the last 24 hours
- alerts by channel
- top risky domains and senders
- campaign volume trend
- model PR-AUC trend
- false-positive and false-negative trend

Primary actions:

- open triage queue
- open top campaign
- switch model version
- filter by time window

### Page 2: Triage Queue

Purpose:

- review suspicious artifacts in priority order

Key content:

- table of alerts with score, channel, timestamp, sender, short preview, and campaign indicator
- filters for channel, score range, status, dataset, and model version
- sort by newest, highest risk, or largest campaign

Primary actions:

- mark as confirmed scam
- mark as benign
- escalate to investigation
- assign owner
- add note

### Page 3: Investigation Workspace

Purpose:

- provide the main analyst workspace for relationship-aware investigation

Layout:

- left rail: investigation list and saved searches
- center pane: selected artifact and timeline
- right pane: linked entities, graph snippet, and model reasons

Key content:

- artifact detail
- sender / recipient / URL / domain cards
- local graph neighborhood
- campaign timeline
- similar or linked artifacts

Primary actions:

- expand graph neighborhood
- pivot to entity profile
- open campaign view
- change alert status

### Page 4: Message Detail

Purpose:

- show one suspicious message or URL in full detail

Key content:

- full text or full URL
- normalized extracted features
- score by model
- relation list
- evidence panel
- historical linked artifacts

Primary actions:

- inspect parsed entities
- compare model outputs
- add label override

### Page 5: Entity Profile

Purpose:

- show risk and behavior for a sender, recipient, URL, or domain

Key content:

- entity summary
- risk over time
- connected artifact count
- channel distribution
- linked campaigns
- first seen / last seen
- related entities

Primary actions:

- open all linked artifacts
- pin entity to graph explorer
- compare related infrastructure

### Page 6: Campaign View

Purpose:

- show a campaign as a cluster rather than as isolated alerts

Key content:

- campaign summary
- time span
- affected channels
- main entities
- graph cluster
- representative artifacts
- analyst notes

Primary actions:

- mark campaign severity
- export campaign report
- open full graph explorer

### Page 7: Graph Explorer

Purpose:

- let users explore the heterogeneous graph directly

Key content:

- zoomable graph
- relation legend
- neighborhood expansion controls
- edge-type filters
- time slider
- side panel with selected node details

Primary actions:

- expand one hop / two hops
- filter to shared sender, recipient, or infrastructure edges
- highlight cross-channel paths
- save graph snapshot

### Page 8: Global Search

Purpose:

- search directly by phone, email, URL, domain, sender ID, recipient ID, or message ID

Key content:

- type-ahead search
- recent searches
- result grouping by object type

Primary actions:

- jump to entity profile
- jump to message detail
- open investigation workspace from search result

### Page 9: Datasets And Ingestion

Purpose:

- expose research datasets and ingestion health

Key content:

- available datasets
- row counts and class balance
- channel coverage
- ingest history
- quality checks
- timestamp coverage
- schema mapping status

Primary actions:

- inspect a dataset
- run validation
- compare dataset versions

### Page 10: Experiments And Models

Purpose:

- compare baselines and GNN runs in one place

Key content:

- experiment table
- PR-AUC, ROC-AUC, precision, recall
- per-channel results
- seed variance
- ablation comparison
- best run summary

Primary actions:

- compare two runs
- open run detail
- inspect error analysis

### Page 11: Error Analysis

Purpose:

- help researchers understand where the model fails

Key content:

- false positives from benign shared infrastructure
- false negatives from first-seen senders
- campaign misses
- channel-specific failure patterns

Primary actions:

- open examples
- filter by channel
- compare baseline vs GNN mistakes

### Page 12: Settings

Purpose:

- workspace, roles, preferences, and model defaults

Key content:

- role and permissions
- default model version
- saved filters
- graph density preferences
- export settings

## 8. Key User Flows

### Flow 1: Triage a suspicious artifact

1. Analyst opens Dashboard.
2. Analyst clicks a high-risk alert.
3. Analyst lands in Triage Queue or Investigation Workspace.
4. Analyst reviews artifact details, reasons, and linked entities.
5. Analyst marks the alert as scam, benign, or escalated.

### Flow 2: Investigate a campaign

1. Investigator opens a suspicious entity or alert.
2. Investigator expands the graph neighborhood.
3. Investigator pivots to Campaign View.
4. Investigator confirms shared infrastructure or shared sender patterns.
5. Investigator exports a campaign summary.

### Flow 3: Compare model runs

1. Researcher opens Experiments And Models.
2. Researcher selects baseline and hetero model runs.
3. Researcher compares PR-AUC overall and by channel.
4. Researcher opens Error Analysis for both runs.
5. Researcher inspects example failures and downloads results.

## 9. V1 Functional Requirements

### Analyst-facing requirements

- show scored alerts from stored experiment outputs or inference jobs
- filter alerts by channel, time, status, and score
- display linked entities and local relationship graph
- support alert status changes and notes

### Investigation requirements

- show sender, recipient, URL, and domain profiles
- show message-to-message shared-edge relationships
- support campaign clustering and timeline view

### Research requirements

- display overall and per-channel metrics
- compare multiple runs side by side
- show seed mean and standard deviation
- surface error-analysis slices from artifacts

### Explainability requirements

- display top reasons or top linked evidence
- show which relation types connect the artifact to risky neighbors
- show raw extracted features for the selected artifact

## 10. Data Visualizations

The designer should plan for:

- metric cards
- stacked channel bars
- timeline charts
- graph network canvas
- node detail drawers
- dense triage tables
- side-by-side experiment comparison panels

## 11. UX And Design Direction

### Overall direction

Desktop-first investigative workspace with a graph-first identity.

### Design principles

- dense but readable
- evidence before decoration
- time and relationships always visible
- cross-channel links visually distinct
- model confidence visible but never isolated from evidence

### Visual guidance

- use a serious operations-tool aesthetic, not a consumer app aesthetic
- support long investigation sessions without visual fatigue
- make node and edge types legible by color and shape, not color alone
- design for split-pane workflows and keyboard-heavy use

## 12. Suggested Navigation

Top-level nav:

- Dashboard
- Triage
- Investigations
- Graph
- Datasets
- Experiments
- Settings

## 13. V2 Opportunities

- live streaming alerts
- annotation workflow and reviewer consensus
- temporal campaign playback
- model explanations at edge and subgraph level
- collaborative investigations
- analyst performance reporting

## 14. Success Metrics

- median time to triage one alert
- analyst precision on reviewed alerts
- number of confirmed campaigns found via graph pivots
- number of cases where cross-channel linkage changes the decision
- researcher time to compare model runs

## 15. Open Questions

- Will the first UI be internal-only or shared with external reviewers?
- Should experiment comparison be read-only in v1 or support launching runs?
- Is campaign detection an automatic backend object in v1, or a derived UI grouping?
- Do we need role-based access in v1, or is one internal workspace enough?
