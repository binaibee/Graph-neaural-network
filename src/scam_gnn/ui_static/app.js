const state = {
  route: "dashboard",
  params: {},
};

const API_FALLBACK_PORT = "8000";

const titleMap = {
  dashboard: ["Dashboard", "Real-time view over scored artifacts, campaigns, and top risky entities."],
  triage: ["Triage Queue", "Filter and review scored alerts from the SQLite-backed investigation store."],
  investigations: ["Investigation Workspace", "Relationship-aware detail view for alerts and linked entities."],
  graph: ["Graph Explorer", "Campaign-centric graph view built from real message and entity relationships."],
  datasets: ["Datasets", "Coverage, class balance, and channel distribution for the active workspace."],
  experiments: ["Experiments", "Compare baselines, homogeneous GNNs, and heterogeneous backends on real data."],
  features: ["Feature Engineering", "How raw text and graph structure are transformed into numbers that models learn from."],
  models: ["Model Ladder", "From simple logistic regression to heterogeneous graph transformers \u2014 architecture, parameters, and results."],
  commands: ["Commands", "Run data pipelines, train models, and rebuild the UI database from the browser."],
  settings: ["Settings", "Configure experiment parameters: dataset, features, splits, ablation, and seeds."],
};

const nodeColors = {
  message: "#89b6ff",
  sender: "#f2a65a",
  recipient: "#8be0c7",
  url: "#d8a5ff",
  domain: "#ff8ea1",
  phone: "#f2a65a",
  email_address: "#8be0c7",
};

const apiState = {
  activeOrigin: null,
  origins: buildApiOrigins(),
};

const TRIAGE_PAGE_SIZE = 50;
const TRIAGE_EXPORT_CAP = 10000;
const triageState = {
  items: [],
  total: 0,
  params: {},
  sort: { key: "predicted_score", dir: "desc" },
};

document.addEventListener("DOMContentLoaded", () => {
  bindNavigation();
  bindSearch();
  renderRoute();
  window.addEventListener("hashchange", renderRoute);
});

async function api(path, options = {}) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const origins = apiState.activeOrigin
    ? [apiState.activeOrigin, ...apiState.origins.filter((origin) => origin !== apiState.activeOrigin)]
    : apiState.origins;
  let lastError = new Error(`Request failed for ${normalizedPath}`);

  for (let index = 0; index < origins.length; index += 1) {
    const origin = origins[index];
    const url = `${origin}${normalizedPath}`;
    let response;
    try {
      response = await fetch(url, options);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      continue;
    }

    const payload = await readJson(response);
    if (response.ok) {
      apiState.activeOrigin = origin;
      return payload;
    }

    if (index < origins.length - 1 && shouldTryNextApiOrigin(origin, response, payload)) {
      continue;
    }

    throw new Error(payload?.error || `Request failed for ${url}`);
  }

  throw buildApiError(lastError, origins);
}

async function apiPost(path, body = {}) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function buildApiOrigins() {
  const origins = [];
  const search = new URLSearchParams(window.location.search);
  addApiOrigin(origins, search.get("apiBase"));
  addApiOrigin(origins, window.SCAM_UI_API_ORIGIN);
  addApiOrigin(origins, document.querySelector('meta[name="scam-api-origin"]')?.content);

  try {
    addApiOrigin(origins, window.localStorage.getItem("scamWorkbenchApiOrigin"));
  } catch (error) {
    // Ignore blocked storage access in preview contexts.
  }

  if (window.location.protocol === "http:" || window.location.protocol === "https:") {
    addApiOrigin(origins, window.location.origin);
  }

  if (isLocalPreview()) {
    addApiOrigin(origins, `http://127.0.0.1:${API_FALLBACK_PORT}`);
    addApiOrigin(origins, `http://localhost:${API_FALLBACK_PORT}`);
  }

  return origins;
}

function addApiOrigin(origins, value) {
  if (!value) {
    return;
  }
  try {
    const origin = new URL(String(value), window.location.href).origin;
    if (origin !== "null" && !origins.includes(origin)) {
      origins.push(origin);
    }
  } catch (error) {
    // Ignore invalid API origin overrides.
  }
}

function isLocalPreview() {
  return window.location.protocol === "file:" || ["127.0.0.1", "localhost"].includes(window.location.hostname);
}

async function readJson(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    return null;
  }
}

function shouldTryNextApiOrigin(origin, response, payload) {
  return (
    origin === window.location.origin
    && isLocalPreview()
    && window.location.port !== API_FALLBACK_PORT
    && (response.status === 404 || payload === null)
  );
}

function buildApiError(error, origins) {
  const message = error instanceof Error ? error.message : String(error);
  if (isLocalPreview() && origins.some((origin) => origin.endsWith(`:${API_FALLBACK_PORT}`)) && /fetch/i.test(message)) {
    return new Error(`Local API unavailable. Start the UI backend on http://127.0.0.1:${API_FALLBACK_PORT} or open the app there directly.`);
  }
  return error instanceof Error ? error : new Error(message);
}

function bindNavigation() {
  document.querySelectorAll(".nav-link").forEach((button) => {
    button.addEventListener("click", () => {
      setRoute(button.dataset.route);
    });
  });
}

function bindSearch() {
  const input = document.getElementById("global-search");
  const results = document.getElementById("search-results");
  let timer = null;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const query = input.value.trim();
    if (!query) {
      results.classList.add("hidden");
      results.innerHTML = "";
      return;
    }
    timer = setTimeout(async () => {
      const payload = await api(`/api/search?q=${encodeURIComponent(query)}`);
      renderSearchResults(payload);
    }, 200);
  });
  document.addEventListener("click", (event) => {
    if (!results.contains(event.target) && event.target !== input) {
      results.classList.add("hidden");
    }
  });
}

function renderSearchResults(payload) {
  const results = document.getElementById("search-results");
  const items = [];
  payload.alerts.forEach((alert) => {
    items.push(`
      <div class="search-item" data-route="investigations" data-alert="${escapeHtml(alert.alert_id)}">
        <strong>${escapeHtml(alert.event_id)}</strong>
        <div class="table-meta">${escapeHtml(alert.channel)} · score ${formatScore(alert.predicted_score)}</div>
        <div>${escapeHtml(alert.preview || "")}</div>
      </div>
    `);
  });
  payload.entities.forEach((entity) => {
    items.push(`
      <div class="search-item" data-route="investigations" data-entity="${escapeHtml(entity.entity_id)}">
        <strong>${escapeHtml(entity.entity_value)}</strong>
        <div class="table-meta">${escapeHtml(entity.entity_type)} · linked ${entity.linked_messages}</div>
      </div>
    `);
  });
  results.innerHTML = items.length ? items.join("") : `<div class="search-item">No results</div>`;
  results.classList.remove("hidden");
  results.querySelectorAll(".search-item[data-route]").forEach((item) => {
    item.addEventListener("click", () => {
      if (item.dataset.alert) {
        setRoute("investigations", { alert: item.dataset.alert });
      } else if (item.dataset.entity) {
        setRoute("investigations", { entity: item.dataset.entity });
      }
      results.classList.add("hidden");
    });
  });
}

function parseHash() {
  const raw = window.location.hash.replace(/^#/, "") || "dashboard";
  const [routeName, queryString] = raw.split("?");
  const route = titleMap[routeName] ? routeName : "dashboard";
  const params = Object.fromEntries(new URLSearchParams(queryString || "").entries());
  return { route, params };
}

function setRoute(route, params = {}) {
  const query = new URLSearchParams(params);
  window.location.hash = query.toString() ? `${route}?${query}` : route;
}

async function renderRoute() {
  const page = document.getElementById("page-content");
  const workspace = document.getElementById("workspace-summary");
  const { route, params } = parseHash();
  state.route = route;
  state.params = params;
  document.getElementById("page-title").textContent = titleMap[route][0];
  document.getElementById("page-subtitle").textContent = titleMap[route][1];
  document.querySelectorAll(".nav-link").forEach((button) => {
    button.classList.toggle("active", button.dataset.route === route);
  });
  page.innerHTML = `<section class="loading-state"><div class="loading-kicker">Loading</div><h2>Fetching live data</h2></section>`;
  try {
    const overview = await api("/api/overview");
    workspace.textContent = `${overview.dataset_name} · ${overview.selected_model}`;
    if (route === "dashboard") {
      const [experiments, features] = await Promise.all([
        api("/api/experiments"),
        api("/api/features").catch(() => null),
      ]);
      page.innerHTML = renderDashboard(overview, experiments, features);
      bindDashboard(page);
      return;
    }
    if (route === "triage") {
      const pageNum = Math.max(1, parseInt(params.page || "1", 10) || 1);
      const offset = (pageNum - 1) * TRIAGE_PAGE_SIZE;
      const query = new URLSearchParams({ ...params, limit: String(TRIAGE_PAGE_SIZE), offset: String(offset) });
      query.delete("page");
      const [alerts, datasets] = await Promise.all([api(`/api/alerts?${query}`), api("/api/datasets")]);
      triageState.items = alerts.items || [];
      triageState.total = alerts.total || 0;
      triageState.params = params;
      page.innerHTML = renderTriage(alerts, datasets, params, pageNum);
      bindTriage(page, pageNum);
      return;
    }
    if (route === "investigations") {
      await renderInvestigation(page, params);
      return;
    }
    if (route === "graph") {
      await renderGraphPage(page, params);
      return;
    }
    if (route === "datasets") {
      const datasets = await api("/api/datasets");
      page.innerHTML = renderDatasets(datasets);
      return;
    }
    if (route === "experiments") {
      const experiments = await api("/api/experiments");
      page.innerHTML = renderExperiments(experiments, overview.error_analysis || []);
      return;
    }
    if (route === "features") {
      const data = await api("/api/features");
      page.innerHTML = renderFeatures(data);
      return;
    }
    if (route === "models") {
      const data = await api("/api/models/info");
      page.innerHTML = renderModels(data);
      return;
    }
    if (route === "commands") {
      const statuses = await api("/api/commands/status").catch(() => ({}));
      const configs = await api("/api/configs").catch(() => []);
      page.innerHTML = renderCommands(statuses, configs);
      bindCommands(page);
      return;
    }
    if (route === "settings") {
      await renderSettingsPage(page);
      return;
    }
  } catch (error) {
    page.innerHTML = `<section class="empty-state"><div class="section-kicker">Error</div><h2>${escapeHtml(error.message)}</h2></section>`;
  }
}

function renderDashboard(overview, experiments, features) {
  const bestExperiments = [...experiments].sort((left, right) => (right.metrics.pr_auc?.mean || 0) - (left.metrics.pr_auc?.mean || 0));
  const scoreBuckets = features?.stats?.score_distribution || [];
  const scoreChart = scoreBuckets.length
    ? svgBarChart(
        [...scoreBuckets].reverse().map((b) => ({
          label: b.bucket,
          value: b.count,
          color: b.bucket >= "0.7" ? "var(--danger)" : b.bucket >= "0.5" ? "var(--warning)" : "var(--primary-strong)",
        })),
        { height: 150 },
      )
    : "";
  return `
    <section class="metric-grid">
      ${metricCard("Artifacts", formatNumber(overview.totals.artifacts), `${formatNumber(overview.totals.test_artifacts)} in test split`)}
      ${metricCard("High Risk", formatNumber(overview.totals.high_risk_artifacts), "score >= 0.90")}
      ${metricCard("Campaigns", formatNumber(overview.totals.campaigns), "connected suspicious clusters")}
      ${metricCard("Best PR-AUC", formatScore(overview.best_model?.pr_auc || 0), overview.best_model ? overview.best_model.model_name : "n/a")}
    </section>
    ${scoreChart ? `<section class="panel" style="margin-bottom:18px;"><div class="section-kicker">Risk distribution</div><h2>Predicted score buckets across all artifacts</h2>${scoreChart}</section>` : ""}
    <section class="dashboard-grid">
      <div class="panel">
        <div class="section-kicker">Channel mix</div>
        <h2>Artifacts by channel</h2>
        <div class="channel-bars">
          ${overview.channels.map((channel) => channelBar(channel.channel, channel.artifact_count, overview.totals.artifacts)).join("")}
        </div>
        <div class="split-grid" style="margin-top:18px;">
          <div class="panel">
            <div class="section-kicker">Trend</div>
            <h3>Daily alert volume</h3>
            ${renderTrendTable(overview.trends)}
          </div>
          <div class="panel">
            <div class="section-kicker">Error analysis</div>
            <h3>Current failure slices</h3>
            <div class="error-list">
              ${overview.error_analysis.map((item) => `<div class="stat-item"><strong>${labelize(item.category)}</strong><div>${formatNumber(item.count)}</div></div>`).join("")}
            </div>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="section-kicker">Top entities</div>
        <h2>Highest-risk linked entities</h2>
        <div class="entity-list">
          ${overview.top_entities.map((entity) => `
            <div class="entity-item" data-entity="${escapeHtml(entity.entity_id)}">
              <strong>${escapeHtml(entity.entity_value)}</strong>
              <div class="table-meta">${escapeHtml(entity.entity_type)} · linked ${entity.linked_messages}</div>
              <div class="score ${scoreClass(entity.risk_mean)}">${formatScore(entity.risk_mean)}</div>
            </div>
          `).join("")}
        </div>
        <div class="section-kicker" style="margin-top:18px;">Campaigns</div>
        <h3>Active suspicious clusters</h3>
        <div class="campaign-list">
          ${overview.top_campaigns.map((campaign) => `
            <div class="campaign-item" data-campaign="${escapeHtml(campaign.campaign_id)}">
              <strong>${escapeHtml(campaign.campaign_id)}</strong>
              <div class="table-meta">${campaign.channels.join(" / ")} · ${campaign.size} artifacts</div>
              <div class="score ${scoreClass(campaign.risk_level)}">${formatScore(campaign.risk_level)}</div>
            </div>
          `).join("")}
        </div>
      </div>
    </section>
    <section class="panel" style="margin-top:18px;">
      <div class="section-kicker">Models</div>
      <h2>Experiment leaderboard</h2>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr><th>Model</th><th>PR-AUC</th><th>ROC-AUC</th><th>Precision</th><th>Recall</th><th>SMS PR-AUC</th><th>URL PR-AUC</th></tr>
          </thead>
          <tbody>
            ${bestExperiments.map((model) => `
              <tr class="table-row">
                <td>${escapeHtml(model.model_name)}</td>
                <td>${formatScore(model.metrics.pr_auc?.mean || 0)}</td>
                <td>${formatScore(model.metrics.roc_auc?.mean || 0)}</td>
                <td>${formatScore(model.metrics.precision?.mean || 0)}</td>
                <td>${formatScore(model.metrics.recall?.mean || 0)}</td>
                <td>${formatScore(model.channels.sms?.pr_auc?.mean || 0)}</td>
                <td>${formatScore(model.channels.url?.pr_auc?.mean || 0)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

const TRIAGE_COLUMNS = [
  { key: "event_id", label: "Alert", cell: (it) => `<td class="mono">${escapeHtml(it.event_id)}</td>` },
  { key: "channel", label: "Channel", cell: (it) => `<td><span class="chip">${escapeHtml(it.channel)}</span></td>` },
  { key: "timestamp", label: "Timestamp", cell: (it) => `<td>${formatDate(it.timestamp)}</td>` },
  { key: "sender_key", label: "Sender", cell: (it) => `<td>${escapeHtml(it.sender_key || "unknown")}</td>` },
  { key: "predicted_score", label: "Score", numeric: true, cell: (it) => `<td><span class="score ${scoreClass(it.predicted_score)}">${formatScore(it.predicted_score)}</span></td>` },
  { key: "campaign_id", label: "Campaign", cell: (it) => `<td>${escapeHtml(it.campaign_id || "single")}</td>` },
  { key: "preview", label: "Preview", cell: (it) => `<td class="cell-preview" title="${escapeHtml(it.preview || "")}">${escapeHtml(it.preview || "")}</td>` },
];

function _triageSortedItems() {
  const { key, dir } = triageState.sort;
  const column = TRIAGE_COLUMNS.find((c) => c.key === key);
  const items = [...triageState.items];
  items.sort((a, b) => {
    let left = a[key];
    let right = b[key];
    if (column?.numeric) {
      left = Number(left) || 0;
      right = Number(right) || 0;
      return dir === "asc" ? left - right : right - left;
    }
    left = String(left ?? "");
    right = String(right ?? "");
    return dir === "asc" ? left.localeCompare(right) : right.localeCompare(left);
  });
  return items;
}

function _triageTableBody() {
  return _triageSortedItems()
    .map(
      (item) => `
      <tr class="table-row" data-alert="${escapeHtml(item.alert_id)}">
        ${TRIAGE_COLUMNS.map((c) => c.cell(item)).join("")}
      </tr>`,
    )
    .join("");
}

function _triageHead() {
  const { key, dir } = triageState.sort;
  return TRIAGE_COLUMNS.map((c) => {
    const indicator = c.key === key ? (dir === "asc" ? " ▲" : " ▼") : "";
    return `<th class="sortable" data-sort="${escapeHtml(c.key)}">${escapeHtml(c.label)}${indicator}</th>`;
  }).join("");
}

function _renderTriagePager(pageNum) {
  const totalPages = Math.max(1, Math.ceil(triageState.total / TRIAGE_PAGE_SIZE));
  const from = triageState.total === 0 ? 0 : (pageNum - 1) * TRIAGE_PAGE_SIZE + 1;
  const to = Math.min(triageState.total, pageNum * TRIAGE_PAGE_SIZE);
  return `
    <div class="pager">
      <button class="action-button" id="triage-prev" ${pageNum <= 1 ? "disabled" : ""}>← Prev</button>
      <span class="pager-info">${formatNumber(from)}–${formatNumber(to)} of ${formatNumber(triageState.total)} · page ${pageNum}/${totalPages}</span>
      <button class="action-button" id="triage-next" ${pageNum >= totalPages ? "disabled" : ""}>Next →</button>
    </div>
  `;
}

function renderTriage(alerts, datasets, params, pageNum = 1) {
  return `
    <section class="panel">
      <div class="filter-row">
        <label><span class="table-meta">Channel</span><select id="triage-channel" class="filter-select"><option value="">All</option>${datasets.channels.map((channel) => `<option value="${escapeHtml(channel.channel)}" ${params.channel === channel.channel ? "selected" : ""}>${escapeHtml(channel.channel)}</option>`).join("")}</select></label>
        <label><span class="table-meta">Split</span><select id="triage-split" class="filter-select"><option value="">All</option>${["train", "validation", "test"].map((split) => `<option value="${split}" ${params.split === split ? "selected" : ""}>${split}</option>`).join("")}</select></label>
        <label><span class="table-meta">Min score</span><input id="triage-score" class="filter-input" type="number" min="0" max="1" step="0.05" value="${escapeHtml(params.min_score || "")}" /></label>
        <label><span class="table-meta">Search</span><input id="triage-query" class="filter-input" type="search" value="${escapeHtml(params.q || "")}" placeholder="sender, recipient, message, url" /></label>
      </div>
      <div class="button-row">
        <button class="action-button primary" id="triage-apply">Apply filters</button>
        <button class="action-button" id="triage-reset">Reset</button>
      </div>
    </section>
    <section class="panel" style="margin-top:18px;">
      <div class="panel-head">
        <div>
          <div class="section-kicker">Alerts</div>
          <h2>${formatNumber(alerts.total)} matching artifacts</h2>
        </div>
        <button class="action-button" id="triage-export">⬇ Export CSV</button>
      </div>
      <div class="table-wrapper">
        <table>
          <thead><tr>${_triageHead()}</tr></thead>
          <tbody id="triage-tbody">${_triageTableBody()}</tbody>
        </table>
      </div>
      ${_renderTriagePager(pageNum)}
    </section>
  `;
}

async function renderInvestigation(page, params) {
  if (params.entity) {
    const payload = await api(`/api/entities/${encodeURIComponent(params.entity)}`);
    page.innerHTML = `
      <section class="investigation-layout">
        <div class="panel">
          <div class="section-kicker">Entity profile</div>
          <h2>${escapeHtml(payload.entity.entity_value)}</h2>
          <div class="pill-row">
            <div class="pill">${escapeHtml(payload.entity.entity_type)}</div>
            <div class="pill">Risk ${formatScore(payload.entity.risk_mean || 0)}</div>
            <div class="pill">Linked messages ${formatNumber(payload.entity.linked_messages)}</div>
          </div>
          <div class="table-wrapper" style="margin-top:16px;">
            <table>
              <thead><tr><th>Alert</th><th>Channel</th><th>Timestamp</th><th>Score</th><th>Campaign</th><th>Preview</th></tr></thead>
              <tbody>
                ${payload.linked_alerts.map((item) => `
                  <tr class="table-row" data-alert="${escapeHtml(item.alert_id)}">
                    <td class="mono">${escapeHtml(item.event_id)}</td>
                    <td>${escapeHtml(item.channel)}</td>
                    <td>${formatDate(item.timestamp)}</td>
                    <td><span class="score ${scoreClass(item.predicted_score)}">${formatScore(item.predicted_score)}</span></td>
                    <td>${escapeHtml(item.campaign_id || "single")}</td>
                    <td>${escapeHtml(item.preview || "")}</td>
                  </tr>`).join("")}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    `;
    bindAlertRows(page);
    return;
  }

  let alertId = params.alert;
  if (!alertId) {
    const fallback = await api("/api/alerts?limit=1");
    alertId = fallback.items[0]?.alert_id;
    if (!alertId) {
      page.innerHTML = `<section class="empty-state"><h2>No alerts available</h2></section>`;
      return;
    }
  }
  const payload = await api(`/api/alerts/${encodeURIComponent(alertId)}`);
  page.innerHTML = `
    <section class="investigation-layout">
      <div class="panel">
        <div class="section-kicker">Artifact</div>
        <h2>${escapeHtml(payload.alert.event_id)}</h2>
        <div class="pill-row">
          <div class="pill">${escapeHtml(payload.alert.channel)}</div>
          <div class="pill">Score ${formatScore(payload.alert.predicted_score)}</div>
          <div class="pill">Status ${escapeHtml(payload.alert.status)}</div>
          <div class="pill">${escapeHtml(payload.alert.campaign_id || "single artifact")}</div>
        </div>
        <p style="margin-top:16px;">${escapeHtml(payload.alert.text_or_url)}</p>
        <div class="button-row" style="margin-top:16px;">
          <button class="action-button" disabled>Confirm scam</button>
          <button class="action-button" disabled>Mark benign</button>
          <button class="action-button" disabled>Escalate</button>
        </div>
        <div class="split-grid" style="margin-top:18px;">
          <div class="panel">
            <div class="section-kicker">Evidence</div>
            <h3>Why it was flagged</h3>
            <div class="stat-list">${payload.reasons.map((reason) => `<div class="stat-item">${escapeHtml(reason)}</div>`).join("")}</div>
          </div>
          <div class="panel">
            <div class="section-kicker">Linked entities</div>
            <h3>Direct neighbors</h3>
            <div class="entity-list">
              ${payload.entities.map((entity) => `
                <div class="entity-item" data-entity="${escapeHtml(entity.entity_id)}">
                  <strong>${escapeHtml(entity.entity_value)}</strong>
                  <div class="table-meta">${escapeHtml(entity.entity_type)} · ${escapeHtml(entity.edge_type)}</div>
                  <div class="score ${scoreClass(entity.risk_mean || 0)}">${formatScore(entity.risk_mean || 0)}</div>
                </div>`).join("")}
            </div>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="section-kicker">Related alerts</div>
        <h3>Connected suspicious messages</h3>
        <div class="stat-list">
          ${payload.related_alerts.map((item) => `
            <div class="entity-item" data-alert="${escapeHtml(item.alert_id)}">
              <strong>${escapeHtml(item.edge_type)}</strong>
              <div class="table-meta">${escapeHtml(item.channel)} · ${formatDate(item.timestamp)}</div>
              <div>${escapeHtml(item.preview || "")}</div>
            </div>`).join("")}
        </div>
      </div>
    </section>
    <section class="panel" style="margin-top:18px;">
      <div class="section-kicker">Graph</div>
      <h2>Local message neighborhood</h2>
      <div class="graph-legend">${renderLegend()}</div>
      <div id="graph-host"></div>
    </section>
  `;
  renderGraph(document.getElementById("graph-host"), payload.graph);
  bindDetailPage(page);
}

async function renderGraphPage(page, params) {
  const campaigns = await api("/api/campaigns");
  if (!campaigns.length) {
    page.innerHTML = `<section class="empty-state"><h2>No campaigns detected for the current model output.</h2></section>`;
    return;
  }
  const activeId = params.campaign || campaigns[0].campaign_id;
  const campaign = await api(`/api/campaigns/${encodeURIComponent(activeId)}`);
  const representative = campaign.alerts[0];
  const detail = representative ? await api(`/api/alerts/${encodeURIComponent(representative.alert_id)}`) : { graph: { nodes: [], edges: [] } };
  page.innerHTML = `
    <section class="triage-layout">
      <div class="panel">
        <div class="section-kicker">Campaigns</div>
        <h2>Suspicious connected components</h2>
        <div class="campaign-list">
          ${campaigns.map((item) => `
            <div class="campaign-item" data-campaign="${escapeHtml(item.campaign_id)}">
              <strong>${escapeHtml(item.campaign_id)}</strong>
              <div class="table-meta">${item.channels.join(" / ")} · ${item.size} artifacts</div>
              <div class="score ${scoreClass(item.risk_level)}">${formatScore(item.risk_level)}</div>
            </div>`).join("")}
        </div>
      </div>
      <div class="panel">
        <div class="section-kicker">Selected campaign</div>
        <h2>${escapeHtml(campaign.campaign_id)}</h2>
        <div class="pill-row">
          <div class="pill">${campaign.channels.join(" / ")}</div>
          <div class="pill">${campaign.size} artifacts</div>
          <div class="pill">Max score ${formatScore(campaign.risk_level)}</div>
        </div>
        <div class="table-wrapper" style="margin-top:16px;">
          <table>
            <thead><tr><th>Alert</th><th>Channel</th><th>Timestamp</th><th>Score</th><th>Preview</th></tr></thead>
            <tbody>
              ${campaign.alerts.map((item) => `
                <tr class="table-row" data-alert="${escapeHtml(item.alert_id)}">
                  <td class="mono">${escapeHtml(item.event_id)}</td>
                  <td>${escapeHtml(item.channel)}</td>
                  <td>${formatDate(item.timestamp)}</td>
                  <td><span class="score ${scoreClass(item.predicted_score)}">${formatScore(item.predicted_score)}</span></td>
                  <td>${escapeHtml(item.preview || "")}</td>
                </tr>`).join("")}
            </tbody>
          </table>
        </div>
      </div>
    </section>
    <section class="panel" style="margin-top:18px;">
      <div class="section-kicker">Campaign graph</div>
      <h2>Representative local neighborhood</h2>
      <div class="graph-legend">${renderLegend()}</div>
      <div id="graph-host"></div>
    </section>
  `;
  renderGraph(document.getElementById("graph-host"), detail.graph);
  page.querySelectorAll("[data-campaign]").forEach((item) => item.addEventListener("click", () => setRoute("graph", { campaign: item.dataset.campaign })));
  bindAlertRows(page);
}

function renderDatasets(datasets) {
  return `
    <section class="metric-grid">
      ${metricCard("Events", formatNumber(datasets.num_events), datasets.config_name)}
      ${metricCard("Entities", formatNumber(datasets.num_entities), "graph nodes")}
      ${metricCard("Relations", formatNumber(datasets.num_relations), "typed edges")}
      ${metricCard("Channels", formatNumber(datasets.channels.length), datasets.track)}
    </section>
    <section class="dataset-grid">
      <div class="panel">
        <div class="section-kicker">Coverage</div>
        <h2>Channel distribution</h2>
        <div class="channel-bars">
          ${datasets.channels.map((channel) => channelBar(channel.channel, channel.artifact_count, datasets.num_events)).join("")}
        </div>
      </div>
      <div class="panel">
        <div class="section-kicker">Class balance</div>
        <h2>Positive-rate summary</h2>
        <div class="stat-list">
          ${datasets.channels.map((channel) => `<div class="stat-item"><strong>${escapeHtml(channel.channel)}</strong><div>${formatPct(channel.positive_rate)}</div></div>`).join("")}
        </div>
      </div>
    </section>
  `;
}

function renderExperiments(experiments, errors) {
  const ranked = [...experiments].sort((a, b) => (b.metrics.pr_auc?.mean || 0) - (a.metrics.pr_auc?.mean || 0));
  const overallBars = ranked.map((m) => ({ label: m.model_name, value: m.metrics.pr_auc?.mean || 0 }));
  const channelGroups = ranked
    .filter((m) => m.channels && (m.channels.sms || m.channels.url))
    .map((m) => ({
      label: m.model_name,
      values: { sms: m.channels.sms?.pr_auc?.mean || 0, url: m.channels.url?.pr_auc?.mean || 0 },
    }));
  return `
    <section class="chart-grid">
      <div class="panel">
        <div class="section-kicker">Visual comparison</div>
        <h2>PR-AUC by model</h2>
        ${svgBarChart(overallBars, { height: 200, valueFormat: formatScore })}
      </div>
      <div class="panel">
        <div class="section-kicker">Per-channel</div>
        <h2>SMS vs URL PR-AUC</h2>
        ${svgGroupedBars(channelGroups, [{ key: "sms", label: "SMS" }, { key: "url", label: "URL" }], { height: 200 })}
      </div>
    </section>
    <section class="panel" style="margin-top:18px;">
      <div class="section-kicker">Leaderboard</div>
      <h2>Overall experiment comparison</h2>
      <div class="table-wrapper">
        <table>
          <thead><tr><th>Model</th><th>PR-AUC</th><th>ROC-AUC</th><th>Precision</th><th>Recall</th><th>Recall@Precision</th></tr></thead>
          <tbody>
            ${experiments.map((model) => `
              <tr>
                <td>${escapeHtml(model.model_name)}</td>
                <td>${formatScore(model.metrics.pr_auc?.mean || 0)}</td>
                <td>${formatScore(model.metrics.roc_auc?.mean || 0)}</td>
                <td>${formatScore(model.metrics.precision?.mean || 0)}</td>
                <td>${formatScore(model.metrics.recall?.mean || 0)}</td>
                <td>${formatScore(model.metrics.recall_at_precision?.mean || 0)}</td>
              </tr>`).join("")}
          </tbody>
        </table>
      </div>
    </section>
    <section class="experiment-grid" style="margin-top:18px;">
      <div class="panel">
        <div class="section-kicker">Per-channel slices</div>
        <h2>SMS and URL comparison</h2>
        <div class="table-wrapper">
          <table>
            <thead><tr><th>Model</th><th>SMS PR-AUC</th><th>SMS Recall</th><th>URL PR-AUC</th><th>URL Recall</th></tr></thead>
            <tbody>
              ${experiments.map((model) => `
                <tr>
                  <td>${escapeHtml(model.model_name)}</td>
                  <td>${formatScore(model.channels.sms?.pr_auc?.mean || 0)}</td>
                  <td>${formatScore(model.channels.sms?.recall?.mean || 0)}</td>
                  <td>${formatScore(model.channels.url?.pr_auc?.mean || 0)}</td>
                  <td>${formatScore(model.channels.url?.recall?.mean || 0)}</td>
                </tr>`).join("")}
            </tbody>
          </table>
        </div>
      </div>
      <div class="panel">
        <div class="section-kicker">Error slices</div>
        <h2>Highest-value failure categories</h2>
        <div class="stat-list">
          ${errors.map((item) => `<div class="stat-item"><strong>${labelize(item.category)}</strong><div>${formatNumber(item.count)} cases</div></div>`).join("")}
        </div>
      </div>
    </section>
  `;
}

function bindDashboard(page) {
  page.querySelectorAll("[data-entity]").forEach((item) => item.addEventListener("click", () => setRoute("investigations", { entity: item.dataset.entity })));
  page.querySelectorAll("[data-campaign]").forEach((item) => item.addEventListener("click", () => setRoute("graph", { campaign: item.dataset.campaign })));
}

function bindTriage(page, pageNum = 1) {
  document.getElementById("triage-apply").addEventListener("click", () => {
    const nextParams = {};
    const channel = document.getElementById("triage-channel").value;
    const split = document.getElementById("triage-split").value;
    const minScore = document.getElementById("triage-score").value;
    const query = document.getElementById("triage-query").value.trim();
    if (channel) nextParams.channel = channel;
    if (split) nextParams.split = split;
    if (minScore) nextParams.min_score = minScore;
    if (query) nextParams.q = query;
    setRoute("triage", nextParams);
  });
  document.getElementById("triage-reset").addEventListener("click", () => setRoute("triage"));

  const prev = document.getElementById("triage-prev");
  const next = document.getElementById("triage-next");
  const baseParams = { ...triageState.params };
  delete baseParams.page;
  if (prev) prev.addEventListener("click", () => setRoute("triage", { ...baseParams, page: String(pageNum - 1) }));
  if (next) next.addEventListener("click", () => setRoute("triage", { ...baseParams, page: String(pageNum + 1) }));

  const exportBtn = document.getElementById("triage-export");
  if (exportBtn) exportBtn.addEventListener("click", () => exportTriageCsv(exportBtn));

  bindTriageSortHeaders(page);
  bindAlertRows(page);
}

function bindTriageSortHeaders(page) {
  page.querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (triageState.sort.key === key) {
        triageState.sort.dir = triageState.sort.dir === "asc" ? "desc" : "asc";
      } else {
        triageState.sort = { key, dir: "desc" };
      }
      const head = page.querySelector("thead tr");
      const body = document.getElementById("triage-tbody");
      if (head) head.innerHTML = _triageHead();
      if (body) body.innerHTML = _triageTableBody();
      bindAlertRows(page);
      bindTriageSortHeaders(page);
    });
  });
}

async function exportTriageCsv(button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Exporting…";
  try {
    const params = { ...triageState.params };
    delete params.page;
    const query = new URLSearchParams({ ...params, limit: String(TRIAGE_EXPORT_CAP), offset: "0" });
    const payload = await api(`/api/alerts?${query}`);
    const header = ["event_id", "channel", "timestamp", "sender_key", "recipient_key", "predicted_score", "predicted_label", "split", "campaign_id", "preview"];
    const rows = [header, ...(payload.items || []).map((it) => header.map((k) => it[k]))];
    downloadCsv("triage_alerts.csv", rows);
    button.textContent = `Exported ${formatNumber((payload.items || []).length)}`;
  } catch (error) {
    button.textContent = "Export failed";
  } finally {
    setTimeout(() => {
      button.disabled = false;
      button.textContent = original;
    }, 2000);
  }
}

function bindAlertRows(page) {
  page.querySelectorAll("[data-alert]").forEach((row) => {
    row.addEventListener("click", () => setRoute("investigations", { alert: row.dataset.alert }));
  });
}

function bindDetailPage(page) {
  bindAlertRows(page);
  page.querySelectorAll("[data-entity]").forEach((item) => item.addEventListener("click", () => setRoute("investigations", { entity: item.dataset.entity })));
}

// ---------------------------------------------------------------------------
// Features page
// ---------------------------------------------------------------------------

const featureCategoryIcons = {
  text_embeddings: "\u2630",
  surface_stats: "\u0023",
  channel_encoding: "\u2261",
  graph_statistics: "\u25ce",
  entity_features: "\u2b22",
  extra_numeric: "\u002b",
};

function renderFeatures(data) {
  const cats = data.categories || [];
  const stats = data.stats || {};

  // Count total features
  const totalFeatures = cats.reduce((sum, cat) => sum + cat.features.length, 0);

  return `
    <section class="metric-grid">
      ${metricCard("Feature Categories", formatNumber(cats.length), "groups of engineered features")}
      ${metricCard("Feature Types", formatNumber(totalFeatures), "individual feature definitions")}
      ${metricCard("Node Types", formatNumber((stats.entity_types || []).length), "in the relationship graph")}
      ${metricCard("Edge Types", formatNumber((stats.edge_types || []).length), "relationship types")}
    </section>

    <!-- Feature pipeline diagram -->
    <section class="panel" style="margin-bottom:18px;">
      <div class="section-kicker">Pipeline</div>
      <h2>Feature extraction flow</h2>
      <div class="pipeline-flow">
        <div class="pipeline-step">
          <div class="pipeline-icon">\ud83d\udcc4</div>
          <div class="pipeline-label">Raw Data</div>
          <div class="table-meta">SMS, Email, URL</div>
        </div>
        <div class="pipeline-arrow">\u2192</div>
        <div class="pipeline-step">
          <div class="pipeline-icon">\ud83d\udd17</div>
          <div class="pipeline-label">Graph Builder</div>
          <div class="table-meta">Nodes + Edges</div>
        </div>
        <div class="pipeline-arrow">\u2192</div>
        <div class="pipeline-step">
          <div class="pipeline-icon">\ud83d\udcdd</div>
          <div class="pipeline-label">Text Encoder</div>
          <div class="table-meta">MiniLM / Hash</div>
        </div>
        <div class="pipeline-arrow">\u2192</div>
        <div class="pipeline-step">
          <div class="pipeline-icon">\ud83d\udcca</div>
          <div class="pipeline-label">Feature Vectors</div>
          <div class="table-meta">Per-message matrix</div>
        </div>
        <div class="pipeline-arrow">\u2192</div>
        <div class="pipeline-step">
          <div class="pipeline-icon">\ud83e\udde0</div>
          <div class="pipeline-label">Models</div>
          <div class="table-meta">Train + Predict</div>
        </div>
      </div>
    </section>

    <!-- Feature categories -->
    <section class="feat-grid">
      ${cats.map((cat) => `
        <div class="panel feat-card">
          <div class="feat-card-header">
            <span class="feat-icon">${featureCategoryIcons[cat.id] || "\u25cf"}</span>
            <div>
              <div class="section-kicker">${escapeHtml(cat.id.replace(/_/g, " "))}</div>
              <h3>${escapeHtml(cat.name)}</h3>
            </div>
          </div>
          <p class="cmd-desc">${escapeHtml(cat.description)}</p>
          <div class="table-wrapper">
            <table>
              <thead><tr><th>Feature</th><th>Type</th><th>Description</th></tr></thead>
              <tbody>
                ${cat.features.map((f) => `
                  <tr>
                    <td class="mono">${escapeHtml(f.name)}</td>
                    <td><span class="feat-type-badge">${escapeHtml(f.type)}</span></td>
                    <td>${escapeHtml(f.description)}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </div>
      `).join("")}
    </section>

    <!-- Graph structure stats -->
    <section class="dashboard-grid" style="margin-top:18px;">
      <div class="panel">
        <div class="section-kicker">Graph structure</div>
        <h2>Node types in the relationship graph</h2>
        <div class="channel-bars">
          ${(stats.entity_types || []).map((et) => channelBar(et.entity_type, et.count, (stats.entity_types || []).reduce((s, e) => s + e.count, 0))).join("")}
        </div>
      </div>
      <div class="panel">
        <div class="section-kicker">Edge types</div>
        <h2>Relationship types and counts</h2>
        <div class="channel-bars">
          ${(stats.edge_types || []).map((et) => channelBar(et.edge_type, et.count, (stats.edge_types || []).reduce((s, e) => s + e.count, 0))).join("")}
        </div>
      </div>
    </section>

    ${(stats.score_distribution || []).length ? `
    <section class="dashboard-grid" style="margin-top:18px;">
      <div class="panel">
        <div class="section-kicker">Score distribution</div>
        <h2>Model output score buckets</h2>
        <div class="channel-bars">
          ${stats.score_distribution.map((b) => channelBar(b.bucket, b.count, stats.score_distribution.reduce((s, x) => s + x.count, 0))).join("")}
        </div>
      </div>
      <div class="panel">
        <div class="section-kicker">High-risk entities</div>
        <h2>Entity types with risk > 0.5</h2>
        ${(stats.high_risk_entity_types || []).length ? `
          <div class="stat-list">
            ${stats.high_risk_entity_types.map((e) => `
              <div class="stat-item">
                <strong>${escapeHtml(e.entity_type)}</strong>
                <div>${formatNumber(e.count)} entities &middot; avg risk ${formatScore(e.avg_risk)}</div>
              </div>
            `).join("")}
          </div>
        ` : `<div class="table-meta">No high-risk entities found.</div>`}
      </div>
    </section>
    ` : ""}
  `;
}

// ---------------------------------------------------------------------------
// Models page
// ---------------------------------------------------------------------------

const tierColors = {
  1: "var(--good)",
  2: "var(--primary)",
  3: "var(--warning)",
  4: "#d8a5ff",
  5: "var(--danger)",
};

function renderModels(data) {
  const models = data.models || [];
  const availability = data.availability || {};
  const metrics = data.metrics || {};

  // Group by tier
  const tiers = {};
  models.forEach((m) => {
    tiers[m.tier] = tiers[m.tier] || { label: m.tier_label, models: [] };
    tiers[m.tier].models.push(m);
  });

  // Dependency status
  const depEntries = Object.entries(availability);

  return `
    <section class="metric-grid">
      ${metricCard("Models", formatNumber(models.length), "in the model ladder")}
      ${metricCard("Tiers", formatNumber(Object.keys(tiers).length), "from baseline to transformer")}
      ${metricCard("Dependencies", `${depEntries.filter(([, v]) => v).length}/${depEntries.length}`, "libraries available")}
      ${metricCard("Best PR-AUC", _bestMetric(metrics, "pr_auc"), _bestModelName(metrics, "pr_auc"))}
    </section>

    <!-- Dependency status -->
    <section class="panel" style="margin-bottom:18px;">
      <div class="section-kicker">Dependencies</div>
      <h2>Library availability</h2>
      <div class="dep-grid">
        ${depEntries.map(([name, available]) => `
          <div class="dep-item ${available ? "dep-available" : "dep-missing"}">
            <span class="dep-dot"></span>
            <span>${escapeHtml(name)}</span>
            <span class="table-meta">${available ? "installed" : "not installed"}</span>
          </div>
        `).join("")}
      </div>
    </section>

    <!-- Model ladder -->
    <section class="model-ladder">
      ${Object.entries(tiers).map(([tier, group]) => `
        <div class="model-tier">
          <div class="tier-header" style="border-left:3px solid ${tierColors[tier] || "var(--muted)"};">
            <div class="section-kicker">Tier ${tier}</div>
            <h2>${escapeHtml(group.label)}</h2>
          </div>
          <div class="model-tier-grid">
            ${group.models.map((m) => {
              const isAvailable = _modelAvailable(m, availability);
              const mMetrics = metrics[m.id] || {};
              return `
                <div class="panel model-card ${isAvailable ? "" : "model-unavailable"}">
                  <div class="model-card-header">
                    <div>
                      <h3>${escapeHtml(m.name)}</h3>
                      <div class="table-meta">${escapeHtml(m.implementation)}</div>
                    </div>
                    <span class="cmd-badge ${isAvailable ? "completed" : "idle"}">${isAvailable ? "Available" : "Unavailable"}</span>
                  </div>
                  <p class="cmd-desc">${escapeHtml(m.description)}</p>

                  <div class="section-kicker" style="margin-top:14px;">Architecture</div>
                  <div class="arch-steps">
                    ${m.architecture.map((step, i) => `
                      <div class="arch-step">
                        <span class="arch-num">${i + 1}</span>
                        <span>${escapeHtml(step)}</span>
                      </div>
                    `).join("")}
                  </div>

                  <div class="section-kicker" style="margin-top:14px;">Parameters</div>
                  <div class="param-grid">
                    ${Object.entries(m.params).map(([k, v]) => `
                      <div class="param-item">
                        <span class="param-key">${escapeHtml(k.replace(/_/g, " "))}</span>
                        <span class="param-val">${escapeHtml(String(v))}</span>
                      </div>
                    `).join("")}
                  </div>

                  ${Object.keys(mMetrics).length ? `
                    <div class="section-kicker" style="margin-top:14px;">Results</div>
                    <div class="param-grid">
                      ${Object.entries(mMetrics).map(([k, v]) => `
                        <div class="param-item">
                          <span class="param-key">${escapeHtml(k.replace(/_/g, " "))}</span>
                          <span class="param-val score ${scoreClass(v.mean)}">${formatScore(v.mean)} <span class="table-meta">\u00b1${formatScore(v.std)}</span></span>
                        </div>
                      `).join("")}
                    </div>
                  ` : ""}

                  <div class="table-meta" style="margin-top:12px;">Requires: ${escapeHtml(m.requires)}</div>
                </div>
              `;
            }).join("")}
          </div>
        </div>
      `).join("")}
    </section>

    <!-- Leaderboard -->
    ${Object.keys(metrics).length ? `
    <section class="panel" style="margin-top:18px;">
      <div class="section-kicker">Leaderboard</div>
      <h2>All models ranked by PR-AUC</h2>
      <div class="table-wrapper">
        <table>
          <thead><tr><th>Rank</th><th>Model</th><th>PR-AUC</th><th>ROC-AUC</th><th>Precision</th><th>Recall</th><th>Recall@Precision</th></tr></thead>
          <tbody>
            ${_rankedModels(metrics).map((m, i) => `
              <tr>
                <td>${i + 1}</td>
                <td>${escapeHtml(m.name)}</td>
                <td class="score ${scoreClass(m.pr_auc)}">${formatScore(m.pr_auc)}</td>
                <td>${formatScore(m.roc_auc)}</td>
                <td>${formatScore(m.precision)}</td>
                <td>${formatScore(m.recall)}</td>
                <td>${formatScore(m.recall_at_precision)}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </section>
    ` : ""}
  `;
}

function _modelAvailable(model, availability) {
  const req = model.requires.toLowerCase();
  if (req === "numpy") return true;
  if (req === "scikit-learn") return !!availability["scikit-learn"];
  if (req.includes("torch")) return !!availability["torch"] && !!availability["torch-geometric"];
  return true;
}

function _bestMetric(metrics, metricName) {
  let best = 0;
  Object.values(metrics).forEach((m) => {
    const val = m[metricName]?.mean || 0;
    if (val > best) best = val;
  });
  return formatScore(best);
}

function _bestModelName(metrics, metricName) {
  let best = 0;
  let name = "n/a";
  Object.entries(metrics).forEach(([model, m]) => {
    const val = m[metricName]?.mean || 0;
    if (val > best) { best = val; name = model; }
  });
  return name;
}

function _rankedModels(metrics) {
  return Object.entries(metrics)
    .map(([name, m]) => ({
      name,
      pr_auc: m.pr_auc?.mean || 0,
      roc_auc: m.roc_auc?.mean || 0,
      precision: m.precision?.mean || 0,
      recall: m.recall?.mean || 0,
      recall_at_precision: m.recall_at_precision?.mean || 0,
    }))
    .sort((a, b) => b.pr_auc - a.pr_auc);
}

// ---------------------------------------------------------------------------
// Commands page
// ---------------------------------------------------------------------------

const commandDefs = [
  {
    id: "generate-synthetic",
    label: "Generate Synthetic Data",
    description: "Create fake multi-channel scam events for quick testing. Generates SMS, email, and URL events with ~35% scam rate.",
    endpoint: "/api/commands/generate-synthetic",
    fields: [
      { name: "output", label: "Output CSV path", type: "text", value: "artifacts/synthetic_events.csv" },
      { name: "seed", label: "Random seed", type: "number", value: "7" },
      { name: "num_events", label: "Number of events", type: "number", value: "96" },
    ],
  },
  {
    id: "prepare-uci-sms",
    label: "Download UCI SMS Dataset",
    description: "Download and standardize the UCI SMS Spam Collection (5,574 real SMS messages).",
    endpoint: "/api/commands/prepare-uci-sms",
    fields: [
      { name: "output", label: "Output CSV path", type: "text", value: "data/public/uci_sms_spam_collection.csv" },
      { name: "force", label: "Force re-download", type: "checkbox", value: false },
    ],
  },
  {
    id: "prepare-phiusiil-url",
    label: "Download PhiUSIIL URL Dataset",
    description: "Download and standardize the UCI PhiUSIIL phishing URL dataset (~34,000 URLs).",
    endpoint: "/api/commands/prepare-phiusiil-url",
    fields: [
      { name: "output", label: "Output CSV path", type: "text", value: "data/public/phiusiil_url_collection.csv" },
      { name: "force", label: "Force re-download", type: "checkbox", value: false },
    ],
  },
  {
    id: "run-research",
    label: "Run Research Experiment",
    description: "Run the full experiment pipeline: load data, build graph, extract features, train all models, evaluate, and save results.",
    endpoint: "/api/commands/run-research",
    fields: [
      { name: "config", label: "Config file", type: "config-select" },
    ],
  },
  {
    id: "build-ui-db",
    label: "Build UI Database",
    description: "Materialize a SQLite database from experiment results for the analyst dashboard.",
    endpoint: "/api/commands/build-ui-db",
    fields: [
      { name: "config", label: "Config file", type: "config-select" },
      { name: "db", label: "Database path", type: "text", value: "artifacts/ui/scam_intelligence.db" },
      { name: "model", label: "Model", type: "text", value: "auto" },
    ],
  },
];

function _statusBadge(status) {
  if (!status || status === "idle") return `<span class="cmd-badge idle">Idle</span>`;
  if (status === "running") return `<span class="cmd-badge running">Running</span>`;
  if (status === "completed") return `<span class="cmd-badge completed">Completed</span>`;
  if (status === "error") return `<span class="cmd-badge error">Error</span>`;
  return `<span class="cmd-badge">${escapeHtml(status)}</span>`;
}

function renderCommands(statuses, configs) {
  const configOptions = configs.map((c) => `<option value="${escapeHtml(c.file)}">${escapeHtml(c.name)}</option>`).join("");

  return `
    <section class="cmd-grid">
      ${commandDefs.map((cmd) => {
        const st = statuses[cmd.id] || {};
        return `
          <div class="panel cmd-card" data-cmd="${cmd.id}">
            <div class="cmd-header">
              <div>
                <div class="section-kicker">Command</div>
                <h3 class="cmd-title">${escapeHtml(cmd.label)}</h3>
              </div>
              ${_statusBadge(st.status)}
            </div>
            <p class="cmd-desc">${escapeHtml(cmd.description)}</p>
            <div class="cmd-fields">
              ${cmd.fields.map((f) => {
                if (f.type === "checkbox") {
                  return `<label class="cmd-field-row"><span class="cmd-field-label">${escapeHtml(f.label)}</span><input type="checkbox" class="cmd-checkbox" data-field="${f.name}" /></label>`;
                }
                if (f.type === "config-select") {
                  return `<label class="cmd-field-row"><span class="cmd-field-label">${escapeHtml(f.label)}</span><select class="filter-select cmd-input" data-field="${f.name}">${configOptions}</select></label>`;
                }
                return `<label class="cmd-field-row"><span class="cmd-field-label">${escapeHtml(f.label)}</span><input type="${f.type}" class="filter-input cmd-input" data-field="${f.name}" value="${escapeHtml(String(f.value || ""))}" /></label>`;
              }).join("")}
            </div>
            <div class="cmd-footer">
              <button class="action-button primary cmd-run" data-cmd="${cmd.id}">Run</button>
              <button class="action-button cmd-refresh" data-cmd="${cmd.id}">Refresh status</button>
            </div>
            ${st.status === "completed" && st.result ? `<div class="cmd-result"><strong>Result:</strong> ${escapeHtml(st.result)}</div>` : ""}
            ${st.status === "error" && st.error ? `<div class="cmd-error"><strong>Error:</strong> ${escapeHtml(st.error)}</div>` : ""}
            ${st.updated_at ? `<div class="table-meta" style="margin-top:8px;">Last updated: ${formatDate(st.updated_at)}</div>` : ""}
          </div>
        `;
      }).join("")}
    </section>
  `;
}

function bindCommands(page) {
  page.querySelectorAll(".cmd-run").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const cmdId = btn.dataset.cmd;
      const cmd = commandDefs.find((c) => c.id === cmdId);
      if (!cmd) return;
      const card = page.querySelector(`.cmd-card[data-cmd="${cmdId}"]`);
      const body = {};
      card.querySelectorAll("[data-field]").forEach((input) => {
        const field = input.dataset.field;
        if (input.type === "checkbox") {
          body[field] = input.checked;
        } else if (input.type === "number") {
          body[field] = Number(input.value);
        } else {
          body[field] = input.value;
        }
      });
      btn.disabled = true;
      btn.textContent = "Starting...";
      try {
        await apiPost(cmd.endpoint, body);
        btn.textContent = "Started";
        // Poll for status
        _pollCommandStatus(cmdId, card, btn);
      } catch (err) {
        btn.textContent = "Run";
        btn.disabled = false;
        alert("Failed to start: " + err.message);
      }
    });
  });

  page.querySelectorAll(".cmd-refresh").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const cmdId = btn.dataset.cmd;
      const card = page.querySelector(`.cmd-card[data-cmd="${cmdId}"]`);
      const runBtn = card.querySelector(".cmd-run");
      await _refreshCommandStatus(cmdId, card, runBtn);
    });
  });
}

async function _refreshCommandStatus(cmdId, card, runBtn) {
  try {
    const st = await api(`/api/commands/status?name=${encodeURIComponent(cmdId)}`);
    const badge = card.querySelector(".cmd-badge");
    if (badge) badge.outerHTML = _statusBadge(st.status);
    // Update result/error
    card.querySelectorAll(".cmd-result, .cmd-error").forEach((el) => el.remove());
    const footer = card.querySelector(".cmd-footer");
    if (st.status === "completed" && st.result) {
      footer.insertAdjacentHTML("afterend", `<div class="cmd-result"><strong>Result:</strong> ${escapeHtml(st.result)}</div>`);
    }
    if (st.status === "error" && st.error) {
      footer.insertAdjacentHTML("afterend", `<div class="cmd-error"><strong>Error:</strong> ${escapeHtml(st.error)}</div>`);
    }
    if (st.status !== "running") {
      runBtn.disabled = false;
      runBtn.textContent = "Run";
    }
  } catch (e) {
    // ignore
  }
}

function _pollCommandStatus(cmdId, card, runBtn) {
  const interval = setInterval(async () => {
    try {
      const st = await api(`/api/commands/status?name=${encodeURIComponent(cmdId)}`);
      const badge = card.querySelector(".cmd-badge");
      if (badge) badge.outerHTML = _statusBadge(st.status);
      if (st.status !== "running") {
        clearInterval(interval);
        await _refreshCommandStatus(cmdId, card, runBtn);
      }
    } catch (e) {
      clearInterval(interval);
      runBtn.disabled = false;
      runBtn.textContent = "Run";
    }
  }, 2000);
}

// ---------------------------------------------------------------------------
// Settings page
// ---------------------------------------------------------------------------

async function renderSettingsPage(page) {
  let configs = [];
  let configData = null;
  let selectedFile = "";

  try {
    configs = await api("/api/configs");
  } catch (e) {
    configs = [];
  }

  if (configs.length > 0) {
    selectedFile = configs[0].file;
    try {
      configData = await api(`/api/config/load?file=${encodeURIComponent(selectedFile)}`);
    } catch (e) {
      configData = null;
    }
  }

  if (!configData) {
    try {
      configData = await api("/api/config/defaults");
    } catch (e) {
      configData = { name: "new_experiment", track: "custom", dataset: {}, features: {}, splits: {}, ablation: {}, seeds: [7, 11, 42], primary_metric: "pr_auc", target_precision: 0.9 };
    }
  }

  page.innerHTML = _renderSettingsForm(configs, selectedFile, configData);
  _bindSettings(page, configs);
}

function _renderSettingsForm(configs, selectedFile, cfg) {
  const ds = cfg.dataset || {};
  const ft = cfg.features || {};
  const sp = cfg.splits || {};
  const ab = cfg.ablation || {};
  const configOptions = configs.map((c) => `<option value="${escapeHtml(c.file)}" ${c.file === selectedFile ? "selected" : ""}>${escapeHtml(c.name)} (${escapeHtml(c.file)})</option>`).join("");

  return `
    <section class="panel" style="margin-bottom:18px;">
      <div class="section-kicker">Config file</div>
      <h3>Load an existing config or create a new one</h3>
      <div class="settings-load-row">
        <select id="settings-file-select" class="filter-select">${configOptions}<option value="__new__">+ New config</option></select>
        <button class="action-button primary" id="settings-load-btn">Load</button>
      </div>
    </section>

    <form id="settings-form" class="settings-grid">
      <!-- General -->
      <section class="panel">
        <div class="section-kicker">General</div>
        <h3>Experiment identity</h3>
        ${_settingsField("name", "Experiment name", "text", cfg.name || "")}
        ${_settingsSelect("track", "Track", ["custom", "benchmark"], cfg.track || "custom")}
        ${_settingsField("primary_metric", "Primary metric", "text", cfg.primary_metric || "pr_auc")}
        ${_settingsField("target_precision", "Target precision", "number", cfg.target_precision ?? 0.9, "0.01")}
        ${_settingsField("seeds", "Seeds (comma-separated)", "text", (cfg.seeds || [7, 11, 42]).join(", "))}
      </section>

      <!-- Dataset -->
      <section class="panel">
        <div class="section-kicker">Dataset</div>
        <h3>Data source configuration</h3>
        ${_settingsSelect("dataset.kind", "Dataset kind", ["synthetic", "uci_sms", "phiusiil_url", "csv"], ds.kind || "synthetic")}
        ${_settingsField("dataset.output_dir", "Output directory", "text", ds.output_dir || "artifacts")}
        ${_settingsField("dataset.sms_path", "SMS CSV path", "text", ds.sms_path || "")}
        ${_settingsField("dataset.email_path", "Email CSV path", "text", ds.email_path || "")}
        ${_settingsField("dataset.url_path", "URL CSV path", "text", ds.url_path || "")}
        ${_settingsField("dataset.max_sms_rows", "Max SMS rows", "number", ds.max_sms_rows ?? "")}
        ${_settingsField("dataset.max_email_rows", "Max email rows", "number", ds.max_email_rows ?? "")}
        ${_settingsField("dataset.max_url_rows", "Max URL rows", "number", ds.max_url_rows ?? "")}
      </section>

      <!-- Features -->
      <section class="panel">
        <div class="section-kicker">Features</div>
        <h3>Feature engineering options</h3>
        ${_settingsField("features.embedding_model", "Embedding model", "text", ft.embedding_model || "all-MiniLM-L6-v2")}
        ${_settingsField("features.fallback_embedding_dim", "Fallback embedding dim", "number", ft.fallback_embedding_dim ?? 24)}
        ${_settingsToggle("features.include_graph_statistics", "Include graph statistics", ft.include_graph_statistics !== false)}
        ${_settingsToggle("features.include_text_hashing_fallback", "Include text hashing fallback", ft.include_text_hashing_fallback !== false)}
      </section>

      <!-- Splits -->
      <section class="panel">
        <div class="section-kicker">Splits</div>
        <h3>Train / Validation / Test ratios</h3>
        ${_settingsField("splits.train_ratio", "Train ratio", "number", sp.train_ratio ?? 0.6, "0.05")}
        ${_settingsField("splits.validation_ratio", "Validation ratio", "number", sp.validation_ratio ?? 0.2, "0.05")}
        ${_settingsField("splits.test_ratio", "Test ratio", "number", sp.test_ratio ?? 0.2, "0.05")}
        <div class="splits-bar" id="splits-bar">
          <div class="splits-segment train" style="width:${((sp.train_ratio ?? 0.6) * 100)}%">Train ${((sp.train_ratio ?? 0.6) * 100).toFixed(0)}%</div>
          <div class="splits-segment val" style="width:${((sp.validation_ratio ?? 0.2) * 100)}%">Val ${((sp.validation_ratio ?? 0.2) * 100).toFixed(0)}%</div>
          <div class="splits-segment test" style="width:${((sp.test_ratio ?? 0.2) * 100)}%">Test ${((sp.test_ratio ?? 0.2) * 100).toFixed(0)}%</div>
        </div>
      </section>

      <!-- Ablation -->
      <section class="panel">
        <div class="section-kicker">Ablation</div>
        <h3>Experimental toggles</h3>
        ${_settingsToggle("ablation.include_cross_channel_edges", "Include cross-channel edges", ab.include_cross_channel_edges !== false)}
        ${_settingsToggle("ablation.include_infrastructure_nodes", "Include infrastructure nodes", ab.include_infrastructure_nodes !== false)}
        <p class="table-meta" style="margin-top:12px;">Disabling cross-channel edges removes links between SMS and URL messages that share senders or infrastructure. Disabling infrastructure nodes removes URL and domain nodes from the graph.</p>
      </section>

      <!-- Actions -->
      <section class="panel settings-actions">
        <div class="section-kicker">Actions</div>
        <h3>Save configuration</h3>
        ${_settingsField("save_path", "Save as", "text", selectedFile || "examples/configs/new_config.json")}
        <div class="button-row" style="margin-top:12px;">
          <button type="submit" class="action-button primary" id="settings-save-btn">Save config</button>
          <button type="button" class="action-button" id="settings-reset-btn">Reset to defaults</button>
        </div>
        <div id="settings-status" class="cmd-result" style="display:none;margin-top:12px;"></div>
      </section>
    </form>
  `;
}

function _settingsField(name, label, type, value, step) {
  const stepAttr = step ? `step="${step}"` : (type === "number" ? 'step="any"' : "");
  return `
    <label class="settings-field">
      <span class="settings-label">${escapeHtml(label)}</span>
      <input type="${type}" class="filter-input settings-input" data-key="${escapeHtml(name)}" value="${escapeHtml(String(value ?? ""))}" ${stepAttr} />
    </label>
  `;
}

function _settingsSelect(name, label, options, current) {
  return `
    <label class="settings-field">
      <span class="settings-label">${escapeHtml(label)}</span>
      <select class="filter-select settings-input" data-key="${escapeHtml(name)}">
        ${options.map((opt) => `<option value="${escapeHtml(opt)}" ${opt === current ? "selected" : ""}>${escapeHtml(opt)}</option>`).join("")}
      </select>
    </label>
  `;
}

function _settingsToggle(name, label, checked) {
  return `
    <label class="settings-field toggle-field">
      <span class="settings-label">${escapeHtml(label)}</span>
      <div class="toggle-switch">
        <input type="checkbox" class="toggle-input settings-input" data-key="${escapeHtml(name)}" ${checked ? "checked" : ""} />
        <span class="toggle-track"><span class="toggle-thumb"></span></span>
      </div>
    </label>
  `;
}

function _bindSettings(page, configs) {
  // Load button
  page.querySelector("#settings-load-btn").addEventListener("click", async () => {
    const select = page.querySelector("#settings-file-select");
    const file = select.value;
    if (file === "__new__") {
      try {
        const defaults = await api("/api/config/defaults");
        const content = page.querySelector("#page-content") || page;
        content.innerHTML = _renderSettingsForm(configs, "", defaults);
        _bindSettings(content, configs);
      } catch (e) {
        alert("Failed to load defaults: " + e.message);
      }
      return;
    }
    try {
      const cfg = await api(`/api/config/load?file=${encodeURIComponent(file)}`);
      const content = page.querySelector("#page-content") || page;
      content.innerHTML = _renderSettingsForm(configs, file, cfg);
      _bindSettings(content, configs);
    } catch (e) {
      alert("Failed to load config: " + e.message);
    }
  });

  // Update splits bar on ratio change
  page.querySelectorAll('[data-key="splits.train_ratio"], [data-key="splits.validation_ratio"], [data-key="splits.test_ratio"]').forEach((input) => {
    input.addEventListener("input", () => {
      const train = parseFloat(page.querySelector('[data-key="splits.train_ratio"]').value) || 0;
      const val = parseFloat(page.querySelector('[data-key="splits.validation_ratio"]').value) || 0;
      const test = parseFloat(page.querySelector('[data-key="splits.test_ratio"]').value) || 0;
      const bar = page.querySelector("#splits-bar");
      if (bar) {
        bar.innerHTML = `
          <div class="splits-segment train" style="width:${(train * 100)}%">Train ${(train * 100).toFixed(0)}%</div>
          <div class="splits-segment val" style="width:${(val * 100)}%">Val ${(val * 100).toFixed(0)}%</div>
          <div class="splits-segment test" style="width:${(test * 100)}%">Test ${(test * 100).toFixed(0)}%</div>
        `;
      }
    });
  });

  // Save button
  page.querySelector("#settings-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const savePath = page.querySelector('[data-key="save_path"]').value;
    if (!savePath) {
      alert("Please enter a file path to save the config.");
      return;
    }

    const config = _collectSettingsForm(page);
    const statusEl = page.querySelector("#settings-status");
    try {
      const result = await apiPost("/api/config/save", { file: savePath, config });
      statusEl.style.display = "block";
      statusEl.textContent = `Saved to ${result.file}`;
      statusEl.className = "cmd-result";
    } catch (err) {
      statusEl.style.display = "block";
      statusEl.textContent = `Error: ${err.message}`;
      statusEl.className = "cmd-error";
    }
  });

  // Reset button
  page.querySelector("#settings-reset-btn").addEventListener("click", async () => {
    try {
      const defaults = await api("/api/config/defaults");
      const content = page.querySelector("#page-content") || page;
      content.innerHTML = _renderSettingsForm(configs, "", defaults);
      _bindSettings(content, configs);
    } catch (e) {
      alert("Failed to load defaults: " + e.message);
    }
  });
}

function _collectSettingsForm(page) {
  const cfg = { dataset: {}, features: {}, splits: {}, ablation: {} };
  page.querySelectorAll(".settings-input").forEach((input) => {
    const key = input.dataset.key;
    if (!key || key === "save_path") return;

    let value;
    if (input.type === "checkbox") {
      value = input.checked;
    } else if (input.type === "number") {
      value = input.value === "" ? null : Number(input.value);
    } else {
      value = input.value;
    }

    // Handle seeds specially
    if (key === "seeds") {
      cfg.seeds = String(value).split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n));
      return;
    }

    // Handle dotted keys like "dataset.kind"
    const parts = key.split(".");
    if (parts.length === 2) {
      cfg[parts[0]][parts[1]] = value;
    } else {
      cfg[key] = value;
    }
  });

  // Convert empty strings to null for optional paths
  ["sms_path", "email_path", "url_path", "fdb_key"].forEach((k) => {
    if (cfg.dataset[k] === "") cfg.dataset[k] = null;
  });

  return cfg;
}

function renderTrendTable(trends) {
  const recent = trends.slice(-12);
  return `
    <div class="table-wrapper">
      <table>
        <thead><tr><th>Day</th><th>Channel</th><th>Artifacts</th></tr></thead>
        <tbody>${recent.map((row) => `<tr><td>${escapeHtml(row.day)}</td><td>${escapeHtml(row.channel)}</td><td>${formatNumber(row.count)}</td></tr>`).join("")}</tbody>
      </table>
    </div>
  `;
}

function renderLegend() {
  return Object.entries(nodeColors)
    .map(([type, color]) => `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${escapeHtml(type)}</span>`)
    .join("");
}

function renderGraph(host, graph) {
  if (!graph || !graph.nodes?.length) {
    host.innerHTML = `<div class="empty-state">No graph neighborhood available.</div>`;
    return;
  }
  const width = 920;
  const height = 460;
  const center = graph.nodes.find((node) => node.type === "message") || graph.nodes[0];
  const others = graph.nodes.filter((node) => node.id !== center.id);
  const positions = new Map();
  positions.set(center.id, { x: width / 2, y: height / 2 });
  others.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(others.length, 1);
    const radius = 150 + (index % 3) * 42;
    positions.set(node.id, {
      x: width / 2 + Math.cos(angle) * radius,
      y: height / 2 + Math.sin(angle) * radius,
    });
  });
  const edges = graph.edges.map((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return "";
    return `<line class="gedge" data-source="${escapeHtml(edge.source)}" data-target="${escapeHtml(edge.target)}" x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}" stroke="#334464" stroke-width="1.4" />`;
  }).join("");
  const nodes = graph.nodes.map((node) => {
    const position = positions.get(node.id);
    const fill = nodeColors[node.type] || "#8fb8ff";
    const isMessage = node.type === "message";
    const nav = isMessage ? `data-nav-alert="${escapeHtml(node.label)}"` : `data-nav-entity="${escapeHtml(node.id)}"`;
    return `
      <g class="gnode" data-id="${escapeHtml(node.id)}" data-type="${escapeHtml(node.type)}" ${nav}>
        <title>${escapeHtml(node.label)} · ${escapeHtml(node.subtitle || node.type)}${node.score != null ? ` · ${formatScore(node.score)}` : ""}</title>
        <circle cx="${position.x}" cy="${position.y}" r="${isMessage ? 24 : 18}" fill="${fill}" fill-opacity="0.18" stroke="${fill}" stroke-width="1.6" />
        <text x="${position.x}" y="${position.y - 30}" text-anchor="middle" fill="#e4ebff" font-size="11">${escapeHtml(shortLabel(node.label))}</text>
        <text x="${position.x}" y="${position.y - 16}" text-anchor="middle" fill="#8fa2c8" font-size="9">${escapeHtml(node.subtitle || node.type)}</text>
      </g>
    `;
  }).join("");
  host.innerHTML = `<svg class="graph-canvas" viewBox="0 0 ${width} ${height}">${edges}${nodes}</svg>`;
  bindGraphInteractions(host);
}

function bindGraphInteractions(host) {
  const svg = host.querySelector("svg");
  if (!svg) return;
  const nodeEls = Array.from(svg.querySelectorAll(".gnode"));
  const edgeEls = Array.from(svg.querySelectorAll(".gedge"));

  const highlight = (id) => {
    const connected = new Set([id]);
    edgeEls.forEach((edge) => {
      if (edge.dataset.source === id || edge.dataset.target === id) {
        edge.classList.add("active");
        connected.add(edge.dataset.source);
        connected.add(edge.dataset.target);
      } else {
        edge.classList.add("dim");
      }
    });
    nodeEls.forEach((node) => {
      node.classList.toggle("dim", !connected.has(node.dataset.id));
    });
  };
  const clearHighlight = () => {
    edgeEls.forEach((edge) => edge.classList.remove("active", "dim"));
    nodeEls.forEach((node) => node.classList.remove("dim"));
  };

  nodeEls.forEach((node) => {
    node.addEventListener("mouseenter", () => highlight(node.dataset.id));
    node.addEventListener("mouseleave", clearHighlight);
    node.addEventListener("click", () => {
      if (node.dataset.navAlert) {
        setRoute("investigations", { alert: node.dataset.navAlert });
      } else if (node.dataset.navEntity) {
        setRoute("investigations", { entity: node.dataset.navEntity });
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Lightweight inline SVG charts (no external dependencies)
// ---------------------------------------------------------------------------

const CHART_PALETTE = ["#4b8fff", "#8be0c7", "#d8a5ff", "#ffb65c", "#ff8ea1", "#66d39a"];

function svgBarChart(items, { height = 200, color = "var(--primary-strong)", valueFormat = formatNumber } = {}) {
  const data = (items || []).filter(Boolean);
  if (!data.length) {
    return `<div class="chart-empty">No data to chart.</div>`;
  }
  const max = Math.max(...data.map((d) => Number(d.value) || 0), 1);
  const bars = data
    .map((d) => {
      const value = Number(d.value) || 0;
      const pct = Math.max(2, (value / max) * 100);
      const tone = d.color || color;
      return `
        <div class="bar-col" title="${escapeHtml(String(d.label))}: ${escapeHtml(valueFormat(value))}">
          <div class="bar-col-value">${escapeHtml(valueFormat(value))}</div>
          <div class="bar-col-track" style="height:${height}px">
            <div class="bar-col-fill" style="height:${pct}%;background:${tone}"></div>
          </div>
          <div class="bar-col-label">${escapeHtml(shortLabel(String(d.label)))}</div>
        </div>`;
    })
    .join("");
  return `<div class="bar-chart">${bars}</div>`;
}

function svgGroupedBars(groups, series, { height = 200, valueFormat = formatScore } = {}) {
  const data = (groups || []).filter(Boolean);
  if (!data.length) {
    return `<div class="chart-empty">No data to chart.</div>`;
  }
  const max = Math.max(
    ...data.flatMap((g) => series.map((s) => Number(g.values[s.key]) || 0)),
    0.0001,
  );
  const legend = series
    .map((s, i) => `<span class="legend-item"><span class="legend-swatch" style="background:${CHART_PALETTE[i % CHART_PALETTE.length]}"></span>${escapeHtml(s.label)}</span>`)
    .join("");
  const cols = data
    .map((g) => {
      const inner = series
        .map((s, i) => {
          const value = Number(g.values[s.key]) || 0;
          const pct = Math.max(2, (value / max) * 100);
          const tone = CHART_PALETTE[i % CHART_PALETTE.length];
          return `<div class="grouped-bar" title="${escapeHtml(s.label)} ${escapeHtml(g.label)}: ${escapeHtml(valueFormat(value))}"><div class="grouped-bar-fill" style="height:${pct}%;background:${tone}"></div></div>`;
        })
        .join("");
      return `
        <div class="grouped-col">
          <div class="grouped-bars" style="height:${height}px">${inner}</div>
          <div class="bar-col-label">${escapeHtml(shortLabel(String(g.label)))}</div>
        </div>`;
    })
    .join("");
  return `<div class="grouped-legend">${legend}</div><div class="grouped-chart">${cols}</div>`;
}

function downloadCsv(filename, rows) {
  const escapeCell = (cell) => {
    const text = String(cell ?? "");
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  const csv = rows.map((row) => row.map(escapeCell).join(",")).join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function metricCard(label, value, meta) {
  return `<section class="metric-card"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${value}</div><div class="metric-meta">${escapeHtml(meta)}</div></section>`;
}

function channelBar(label, value, total) {
  const width = total ? Math.max(6, (value / total) * 100) : 0;
  return `<div class="bar-row"><div>${escapeHtml(label)}</div><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><div>${formatNumber(value)}</div></div>`;
}

function shortLabel(value) {
  return String(value || "").length > 18 ? `${String(value).slice(0, 18)}…` : String(value || "");
}

function scoreClass(value) {
  if (value >= 0.9) return "high";
  if (value >= 0.6) return "medium";
  return "low";
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value || 0);
}

function formatScore(value) {
  return Number(value || 0).toFixed(3);
}

function formatPct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function labelize(value) {
  return String(value).replaceAll("_", " ");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
