const state = {
  data: null,
  view: "dynamic",
  sort: "resolve",
  query: "",
  kind: "all"
};

const els = {
  noticeBar: document.querySelector("#noticeBar"),
  projectLinks: document.querySelector("#projectLinks"),
  runStats: document.querySelector("#runStats"),
  kindChips: document.querySelector("#kindChips"),
  routerSearch: document.querySelector("#routerSearch"),
  pageTitle: document.querySelector("#pageTitle"),
  heroStats: document.querySelector("#heroStats"),
  pickGrid: document.querySelector("#pickGrid"),
  copyLink: document.querySelector("#copyLink"),
  leaderboardRows: document.querySelector("#leaderboardRows"),
  chartPanel: document.querySelector("#chartPanel"),
  dynamicDescription: document.querySelector("#dynamicDescription"),
  staticDescription: document.querySelector("#staticDescription"),
  versionPill: document.querySelector("#versionPill"),
  workloadGrid: document.querySelector("#workloadGrid"),
  splitRule: document.querySelector("#splitRule"),
  splitStats: document.querySelector("#splitStats")
};

function pct(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function usd(value, digits = 3) {
  return `$${Number(value).toFixed(digits)}`;
}

function number(value, digits = 0) {
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits
  });
}

function valueScore(router) {
  if (!router.avg_cost_usd) return 0;
  return (router.resolve_rate * 100) / router.avg_cost_usd;
}

function classToken(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function sortRouters(rows, sortKey = state.sort) {
  const comparators = {
    value: (a, b) => valueScore(b) - valueScore(a) || b.resolve_rate - a.resolve_rate,
    resolve: (a, b) => b.resolve_rate - a.resolve_rate || a.avg_cost_usd - b.avg_cost_usd,
    cost: (a, b) => a.avg_cost_usd - b.avg_cost_usd || b.resolve_rate - a.resolve_rate,
    steps: (a, b) => a.avg_steps - b.avg_steps || b.resolve_rate - a.resolve_rate
  };
  return [...rows].sort(comparators[sortKey] || comparators.resolve);
}

function filteredRouters() {
  const q = state.query.trim().toLowerCase();
  return state.data.dynamic.routers.filter((router) => {
    const matchesKind = state.kind === "all" || router.kind === state.kind;
    const haystack = [
      router.name,
      router.kind,
      router.benchmark,
      router.notes,
      ...(router.badges || [])
    ]
      .join(" ")
      .toLowerCase();
    return matchesKind && (!q || haystack.includes(q));
  });
}

function renderNotice() {
  els.noticeBar.innerHTML = `
    <div class="notice-inner">
      <span class="notice-spark" aria-hidden="true"></span>
      <strong>Leaderboard preview</strong>
      <span>${state.data.notice}</span>
      <a href="${state.data.links.github}" target="_blank" rel="noreferrer">View repository</a>
    </div>
  `;
}

function renderLinks() {
  const links = state.data.links;
  const items = [
    ["About", links.github],
    ["Submit", links.submission],
    ["Dataset", links.dataset],
    ["Manifest", links.static_manifest],
    ["Paper", links.paper],
    ["Croissant", links.croissant]
  ].filter(([, href]) => href);

  els.projectLinks.innerHTML = items
    .map(([label, href]) => `<a href="${href}" target="_blank" rel="noreferrer">${label}</a>`)
    .join("");
}

function renderRunStats() {
  const routers = state.data.dynamic.routers;
  const stats = [
    [`${routers.length}`, "routers"],
    [`${state.data.dynamic.split.cases}`, "cases"],
    [`${state.data.static.total_rows}`, "static labels"]
  ];
  els.runStats.innerHTML = stats
    .map(([value, label]) => `<span><strong>${value}</strong> ${label}</span>`)
    .join("");
}

function renderKindChips() {
  const kinds = ["all", ...Array.from(new Set(state.data.dynamic.routers.map((r) => r.kind))).sort()];
  els.kindChips.innerHTML = `
    <span class="chip-label">Router type:</span>
    ${kinds
      .map((kind) => {
        const label = kind === "all" ? "All" : kind;
        const active = state.kind === kind ? "is-active" : "";
        return `<button class="${active}" type="button" data-kind="${kind}">${label}</button>`;
      })
      .join("")}
  `;

  els.kindChips.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.kind = button.dataset.kind;
      renderKindChips();
      renderDynamic();
    });
  });
}

function renderHeroStats() {
  const routers = state.data.dynamic.routers;
  const bestResolve = sortRouters(routers, "resolve")[0];
  const bestCost = sortRouters(routers, "cost")[0];
  const bestValue = sortRouters(routers, "value")[0];
  const stats = [
    ["Best resolve", pct(bestResolve.resolve_rate), bestResolve.name],
    ["Lowest avg cost", usd(bestCost.avg_cost_usd), bestCost.name],
    ["Best value", number(valueScore(bestValue), 1), bestValue.name],
    ["Static rows", number(state.data.static.total_rows), `${state.data.static.workloads.length} workloads`]
  ];

  els.heroStats.innerHTML = stats
    .map(
      ([label, value, note]) => `
        <article>
          <span>${label}</span>
          <strong>${value}</strong>
          <small>${note}</small>
        </article>
      `
    )
    .join("");
}

function renderQuickPicks() {
  const routers = state.data.dynamic.routers;
  const picks = [
    {
      key: "overall",
      label: "Best Overall",
      metric: "Resolve",
      value: pct(sortRouters(routers, "resolve")[0].resolve_rate),
      router: sortRouters(routers, "resolve")[0],
      description: "Highest resolved-case rate, tie-broken by lower average cost.",
      sort: "resolve",
      icon: "trophy"
    },
    {
      key: "budget",
      label: "Best Budget",
      metric: "Avg Cost",
      value: usd(sortRouters(routers, "cost")[0].avg_cost_usd),
      router: sortRouters(routers, "cost")[0],
      description: "Lowest average routed spend on the heldout dynamic split.",
      sort: "cost",
      icon: "dollar"
    },
    {
      key: "value",
      label: "Best Value",
      metric: "Value",
      value: number(valueScore(sortRouters(routers, "value")[0]), 1),
      router: sortRouters(routers, "value")[0],
      description: "Best resolve-rate percentage per average dollar.",
      sort: "value",
      icon: "gem"
    },
    {
      key: "steps",
      label: "Fewest Steps",
      metric: "Avg Steps",
      value: number(sortRouters(routers, "steps")[0].avg_steps, 1),
      router: sortRouters(routers, "steps")[0],
      description: "Shortest average routed trace among evaluated routers.",
      sort: "steps",
      icon: "zap"
    }
  ];

  els.pickGrid.innerHTML = picks
    .map(
      (pick) => `
        <button class="pick-card" type="button" data-pick-sort="${pick.sort}">
          <span class="pick-top">
            <span class="pick-label">
              <span class="icon ${pick.icon}" aria-hidden="true"></span>
              ${pick.label}
            </span>
            <span class="pick-score">${pick.value}</span>
          </span>
          <strong>${pick.router.name}</strong>
          <span class="pick-meta">${pick.metric} · ${pick.router.resolved}/${pick.router.cases} cases · ${usd(pick.router.avg_cost_usd)}</span>
          <span class="pick-desc">${pick.description}</span>
        </button>
      `
    )
    .join("");

  els.pickGrid.querySelectorAll("[data-pick-sort]").forEach((button) => {
    button.addEventListener("click", () => setDynamicSort(button.dataset.pickSort));
  });
}

function renderChart(rows) {
  const maxRate = Math.max(...rows.map((r) => r.resolve_rate), 1);
  els.chartPanel.innerHTML = rows
    .map((router) => {
      const width = (router.resolve_rate / maxRate) * 100;
      return `
        <div class="chart-row">
          <span>${router.name}</span>
          <div class="chart-track"><i style="width: ${width}%"></i></div>
          <strong>${pct(router.resolve_rate)}</strong>
        </div>
      `;
    })
    .join("");
}

function renderDynamic() {
  const rows = sortRouters(filteredRouters());
  els.dynamicDescription.textContent = state.data.dynamic.description;
  els.versionPill.textContent = state.data.version;
  renderChart(rows);
  els.leaderboardRows.innerHTML = rows
    .map((router, index) => {
      const badges = (router.badges || [router.kind])
        .map((badge) => `<span class="badge">${badge}</span>`)
        .join("");
      return `
        <tr>
          <td><span class="rank">${index + 1}</span></td>
          <td>
            <div class="router-title">${router.name}</div>
            <div class="router-sub">${router.kind} · ${router.benchmark}</div>
          </td>
          <td><div class="badge-list">${badges}</div></td>
          <td>
            <div class="strong-number">${pct(router.resolve_rate)}</div>
            <div class="mini-bar" aria-hidden="true"><span style="width: ${router.resolve_rate * 100}%"></span></div>
            <div class="router-sub">${router.resolved}/${router.cases} resolved</div>
          </td>
          <td>
            <div class="strong-number">${usd(router.avg_cost_usd)}</div>
            <div class="router-sub">${usd(router.total_cost_usd)} total</div>
          </td>
          <td>
            <div class="strong-number">${number(router.avg_steps, 1)}</div>
            <div class="router-sub">${number(router.total_steps)} total</div>
          </td>
          <td><div class="strong-number">${number(valueScore(router), 1)}</div></td>
          <td class="notes">${router.notes || ""}</td>
        </tr>
      `;
    })
    .join("");
}

function renderStatic() {
  const total = state.data.static.total_rows;
  els.staticDescription.textContent = `${number(total)} conditional per-call labels across ${state.data.static.workloads.length} workloads.`;
  els.workloadGrid.innerHTML = state.data.static.workloads
    .map((workload) => {
      const share = workload.rows / total;
      return `
        <article class="workload-card">
          <div>
            <h3>${workload.name}</h3>
            <p>${pct(share)} of static rows</p>
          </div>
          <strong>${number(workload.rows)}</strong>
          <div class="mini-bar" aria-hidden="true"><span style="width: ${share * 100}%"></span></div>
          <dl>
            <dt>Stage</dt>
            <dd>${workload.stage || "n/a"}</dd>
            <dt>Version</dt>
            <dd>${workload.version || "n/a"}</dd>
          </dl>
        </article>
      `;
    })
    .join("");
}

function renderProtocol() {
  const split = state.data.dynamic.split;
  els.splitRule.textContent = split.rule;
  els.splitStats.innerHTML = Object.entries(split)
    .filter(([key]) => key !== "rule")
    .map(
      ([key, value]) => `
        <dt>${key.replaceAll("_", " ")}</dt>
        <dd>${number(value)}</dd>
      `
    )
    .join("");
}

function setDynamicSort(sort) {
  state.sort = sort;
  state.view = "dynamic";
  const active = document.querySelector(`[data-view="dynamic"][data-sort="${sort}"]`);
  updateTabs(active);
  switchView("dynamic");
  renderDynamic();
}

function switchView(view) {
  state.view = view;
  document.querySelectorAll(".view").forEach((section) => {
    section.classList.toggle("is-active", section.id === `${view}View`);
  });
  document.body.dataset.view = view;
}

function updateTabs(activeButton) {
  document.querySelectorAll("#metricTabs button").forEach((button) => {
    button.classList.toggle("is-active", button === activeButton);
  });
  if (activeButton?.dataset.title) {
    els.pageTitle.textContent = activeButton.dataset.title;
    document.title = `${activeButton.dataset.title} | TwinRouterBench`;
  }
}

function bindEvents() {
  document.querySelectorAll("#metricTabs button").forEach((button) => {
    button.addEventListener("click", () => {
      updateTabs(button);
      if (button.dataset.view === "dynamic") {
        state.sort = button.dataset.sort;
        switchView("dynamic");
        renderDynamic();
      } else {
        switchView(button.dataset.view);
      }
    });
  });

  els.routerSearch.addEventListener("input", (event) => {
    state.query = event.target.value;
    renderDynamic();
  });

  els.copyLink.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      els.copyLink.classList.add("copied");
      els.copyLink.querySelector("span:last-child").textContent = "Copied";
      setTimeout(() => {
        els.copyLink.classList.remove("copied");
        els.copyLink.querySelector("span:last-child").textContent = "Copy link";
      }, 1200);
    } catch {
      window.location.hash = "leaderboard";
    }
  });
}

async function init() {
  const response = await fetch("data/leaderboard.json");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  state.data = await response.json();
  renderNotice();
  renderLinks();
  renderRunStats();
  renderKindChips();
  renderHeroStats();
  renderQuickPicks();
  renderDynamic();
  renderStatic();
  renderProtocol();
  bindEvents();
}

init().catch((error) => {
  document.body.innerHTML = `<main class="page error"><h1>Unable to load leaderboard data</h1><p>${error}</p></main>`;
});
