/* MTG Deck Builder frontend.
 *
 * The deck-building engine is the Python package in mtg_deckbuilder/, run
 * in-browser via Pyodide. The browser's only other job is looking up the
 * pasted card names on Scryfall (batched, cached in localStorage) and
 * rendering the JSON that webapi.run_json returns.
 */
"use strict";

const ENGINE_FILES = [
  "__init__.py", "carddata.py", "collection.py", "tags.py",
  "synergy.py", "deckbuilder.py", "webapi.py",
];
const SCRYFALL_COLLECTION_URL = "https://api.scryfall.com/cards/collection";
const SCRYFALL_NAMED_URL = "https://api.scryfall.com/cards/named";
const CHUNK = 75;                       // Scryfall's per-request identifier cap
const CACHE_PREFIX = "mtgdeck:v2:";     // v2: card images added to cache entries
const CACHE_TTL_MS = 7 * 24 * 3600 * 1000;

// Only the fields the engine reads; keeps localStorage and the JS<->Python
// boundary small (full Scryfall objects are ~8x bigger).
const CARD_FIELDS = [
  "name", "layout", "mana_cost", "cmc", "colors", "color_identity",
  "type_line", "oracle_text", "keywords", "power", "toughness",
  "produced_mana", "rarity",
];
const FACE_FIELDS = ["name", "mana_cost", "type_line", "oracle_text", "power", "toughness"];

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const resultsEl = $("results");

let api = null;                          // Pyodide module proxy for webapi

// ---------------------------------------------------------------------------
// Remember the collection + options across refreshes (stays in this browser).
// ---------------------------------------------------------------------------

const UI_STATE_KEY = "mtgdeck:ui";
let restoredCollection = false;

function saveUiState() {
  try {
    localStorage.setItem(UI_STATE_KEY, JSON.stringify({
      collection: $("collection").value,
      format: $("format").value,
      colors: $("colors").value,
      theme: $("theme").value,
      commander: $("commander").value,
      playsets: $("playsets").checked,
    }));
  } catch { /* storage full or blocked — nothing to do */ }
}

function loadUiState() {
  try {
    return JSON.parse(localStorage.getItem(UI_STATE_KEY)) ?? {};
  } catch { return {}; }
}

function restoreUiState() {
  const state = loadUiState();
  if (state.collection) {
    $("collection").value = state.collection;
    restoredCollection = true;
  }
  if (state.format) {
    $("format").value = state.format;
    updateFormatVisibility();
  }
  if (state.colors) $("colors").value = state.colors;
  if (state.commander) $("commander").value = state.commander;
  $("playsets").checked = !!state.playsets;
  // The theme select's options are filled once the engine loads; boot()
  // re-applies the saved theme after that.
}

// ---------------------------------------------------------------------------
// Engine boot
// ---------------------------------------------------------------------------

async function boot() {
  try {
    setStatus("Loading deck-building engine…");
    const pyodide = await loadPyodide();
    setStatus("Loading deck-building engine… (card logic)");
    pyodide.FS.mkdir("mtg_deckbuilder");
    await Promise.all(ENGINE_FILES.map(async (f) => {
      const res = await fetch(`mtg_deckbuilder/${f}`);
      if (!res.ok) throw new Error(`could not fetch mtg_deckbuilder/${f}`);
      pyodide.FS.writeFile(`mtg_deckbuilder/${f}`, await res.text());
    }));
    pyodide.runPython("import sys; sys.path.insert(0, '.')");
    api = pyodide.pyimport("mtg_deckbuilder.webapi");

    const meta = JSON.parse(api.run_json(JSON.stringify({ action: "meta" })));
    const themeSelect = $("theme");
    for (const t of meta.result.themes) {
      const opt = document.createElement("option");
      opt.value = t.key;
      opt.textContent = t.name;
      themeSelect.appendChild(opt);
    }

    const savedTheme = loadUiState().theme;
    if (savedTheme) themeSelect.value = savedTheme;

    for (const id of ["btn-build", "btn-suggest", "btn-analyze"]) $(id).disabled = false;
    setStatus(restoredCollection
      ? "Ready. Your collection was restored from last time."
      : "Ready. Paste your collection and build a deck.", "ok");
  } catch (err) {
    setStatus(`The engine failed to load: ${err.message ?? err}. ` +
      "Try reloading the page.", "error");
  }
}

// ---------------------------------------------------------------------------
// Scryfall lookups
// ---------------------------------------------------------------------------

function cacheGet(name) {
  try {
    const raw = localStorage.getItem(CACHE_PREFIX + name.toLowerCase());
    if (!raw) return null;
    const { t, c } = JSON.parse(raw);
    if (Date.now() - t > CACHE_TTL_MS) return null;
    return c;
  } catch { return null; }
}

function cachePut(name, card) {
  try {
    localStorage.setItem(CACHE_PREFIX + name.toLowerCase(),
      JSON.stringify({ t: Date.now(), c: card }));
  } catch { /* cache full — fine, lookups still work */ }
}

function slimCard(card) {
  const out = {};
  for (const f of CARD_FIELDS) if (card[f] !== undefined) out[f] = card[f];
  if (card.image_uris?.normal) out.images = [card.image_uris.normal];
  if (Array.isArray(card.card_faces)) {
    out.card_faces = card.card_faces.map((face) => {
      const slim = {};
      for (const f of FACE_FIELDS) if (face[f] !== undefined) slim[f] = face[f];
      return slim;
    });
    if (!out.images) {
      const faces = card.card_faces
        .map((face) => face.image_uris?.normal)
        .filter(Boolean);
      if (faces.length) out.images = faces;
    }
  }
  return out;
}

// name (normalized) -> [image urls]; filled from every Scryfall response.
const imageIndex = new Map();

function indexImages(cards) {
  for (const card of cards) {
    if (!card.images) continue;
    imageIndex.set(normalize(card.name), card.images);
    if (card.name.includes(" // ")) {
      imageIndex.set(normalize(frontFace(card.name)), card.images);
    }
  }
}

const normalize = (s) => s.replace(/\s+/g, " ").trim().toLowerCase();
const frontFace = (s) => s.split(" // ")[0].trim();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function scryfallChunk(names) {
  const res = await fetch(SCRYFALL_COLLECTION_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ identifiers: names.map((name) => ({ name })) }),
  });
  if (!res.ok) {
    throw new Error(`Scryfall lookup failed (HTTP ${res.status}) — ` +
      "wait a moment and try again");
  }
  const data = await res.json();
  return {
    cards: (data.data ?? []).map(slimCard),
    notFound: (data.not_found ?? []).map((idn) => idn.name),
  };
}

/** Look up card objects for the given names. Returns {cards, notFound}. */
async function lookupCards(names) {
  const cards = [];
  const byKey = new Map();  // normalized requested-name -> still missing?
  const misses = [];
  for (const name of names) {
    const cached = cacheGet(name);
    if (cached) cards.push(cached);
    else misses.push(name);
  }

  let notFound = [];
  for (let i = 0; i < misses.length; i += CHUNK) {
    setStatus(`Looking up cards on Scryfall… ${Math.min(i + CHUNK, misses.length)}` +
      `/${misses.length} (${names.length - misses.length} already cached)`);
    const chunk = misses.slice(i, i + CHUNK);
    const result = await scryfallChunk(chunk);
    cards.push(...result.cards);
    notFound.push(...result.notFound);
    if (i + CHUNK < misses.length) await sleep(120);
  }

  // Full double-faced names sometimes miss; retry those with the front face.
  const retry = notFound.filter((n) => n.includes(" // "));
  if (retry.length) {
    notFound = notFound.filter((n) => !n.includes(" // "));
    const result = await scryfallChunk(retry.map(frontFace));
    cards.push(...result.cards);
    notFound.push(...result.notFound);
  }

  // Write fresh results back to the cache under the requested name.
  for (const card of cards) {
    byKey.set(normalize(card.name), card);
    if (card.name.includes(" // ")) byKey.set(normalize(frontFace(card.name)), card);
  }
  for (const name of misses) {
    const hit = byKey.get(normalize(name)) ?? byKey.get(normalize(frontFace(name)));
    if (hit) cachePut(name, hit);
  }
  indexImages(cards);
  return { cards, notFound };
}

/** Image URLs for one card, fetching from Scryfall if we don't have them. */
async function imagesFor(name) {
  const key = normalize(name);
  if (imageIndex.has(key)) return imageIndex.get(key);
  const res = await fetch(
    `${SCRYFALL_NAMED_URL}?exact=${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`Scryfall doesn't know “${name}” (HTTP ${res.status})`);
  const card = slimCard(await res.json());
  indexImages([card]);
  return imageIndex.get(key) ?? card.images ?? [];
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

async function runAction(action) {
  const text = $("collection").value;
  if (!text.trim()) {
    setStatus("Paste your collection first (or press “Try a sample”).", "error");
    return;
  }
  const buttons = ["btn-build", "btn-suggest", "btn-analyze"].map($);
  buttons.forEach((b) => (b.disabled = true));
  try {
    const names = JSON.parse(api.collection_names(text));
    if (!names.length) throw new Error("no card names found in the text");
    const { cards, notFound } = await lookupCards(names);

    setStatus(action === "build" ? "Building the deck…" :
      action === "suggest" ? "Searching for decks…" : "Analyzing…");
    const request = { action, collection: text, cards };
    if (action !== "analyze") {
      request.format = $("format").value;
      request.playsets = $("playsets").checked;
    }
    if (action === "build") {
      request.colors = $("colors").value.trim();
      request.theme = $("theme").value;
      request.commander = $("format").value === "commander" ? $("commander").value : "";
    }
    const response = JSON.parse(api.run_json(JSON.stringify(request)));
    if (!response.ok) throw new Error(response.error);

    const render = { build: renderDeck, suggest: renderIdeas, analyze: renderAnalysis }[action];
    render(response.result, notFound);
    resultsEl.hidden = false;
    resultsEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setStatus("Done.", "ok");
  } catch (err) {
    setStatus(`${err.message ?? err}`, "error");
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function setStatus(message, kind) {
  statusEl.textContent = message;
  statusEl.className = "status" + (kind ? ` ${kind}` : "");
}

const escapeHtml = (s) => s.replace(/[&<>"']/g, (ch) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

function manaPips(cost) {
  if (!cost) return "";
  const pips = [...cost.matchAll(/\{([^}]+)\}/g)].map(([, sym]) => {
    const color = sym[0].toUpperCase();
    const cls = "WUBRG".includes(color) ? ` ${color}` : "";
    return `<span class="pip${cls}">${escapeHtml(sym.replace(/\//g, ""))}</span>`;
  });
  return `<span class="mana">${pips.join("")}</span>`;
}

const colorChips = (label) =>
  `<span class="color-chips">${[...label].map((c) =>
    `<span class="pip ${c}">${c}</span>`).join("")}</span>`;

/** A clickable card name that pops up the card image. */
const cardLink = (name) =>
  `<span class="card-link" role="button" tabindex="0" ` +
  `data-card="${escapeHtml(name)}">${escapeHtml(name)}</span>`;

// ---- card image modal ----

function closeCardModal() {
  $("card-modal").hidden = true;
}

async function showCardModal(name) {
  const modal = $("card-modal");
  const imagesEl = $("card-modal-images");
  const caption = $("card-modal-caption");
  modal.hidden = false;
  imagesEl.innerHTML = "";
  caption.textContent = `Loading ${name}…`;
  try {
    const images = await imagesFor(name);
    if (!images.length) throw new Error(`no image available for “${name}”`);
    if (modal.hidden) return;              // closed while loading
    imagesEl.innerHTML = images.map((url) =>
      `<img src="${escapeHtml(url)}" alt="${escapeHtml(name)}">`).join("");
    caption.textContent = name;
  } catch (err) {
    caption.textContent = `${err.message ?? err}`;
  }
}

document.addEventListener("click", (event) => {
  const link = event.target.closest(".card-link");
  if (link) { showCardModal(link.dataset.card); return; }
  if (event.target.closest("#card-modal-close") ||
      event.target.id === "card-modal-backdrop") closeCardModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeCardModal();
  if (event.key === "Enter" && event.target.classList?.contains("card-link")) {
    showCardModal(event.target.dataset.card);
  }
});

function missingNote(missing, notFound) {
  const all = [...new Set([...(missing ?? []), ...(notFound ?? [])])];
  if (!all.length) return "";
  const shown = all.slice(0, 8).map(escapeHtml).join(", ");
  return `<p class="warn">Not found on Scryfall (skipped): ${shown}` +
    `${all.length > 8 ? ` … and ${all.length - 8} more` : ""}</p>`;
}

function renderDeck(deck, notFound) {
  const title = deck.commander
    ? `Commander — ${escapeHtml(deck.theme_name)}`
    : `60-card — ${escapeHtml(deck.theme_name)}`;
  const categories = deck.categories.map((cat) => `
    <div class="category">
      <h3>${escapeHtml(cat.name)} <span class="count">(${cat.count})</span></h3>
      <ul>${cat.cards.map((card) => `
        <li title="${escapeHtml(card.reasons.join("; "))}">
          <span class="card-count">${card.count}</span>
          <span class="card-name">${cardLink(card.name)}</span>
          ${manaPips(card.mana_cost)}
        </li>`).join("")}
      </ul>
    </div>`).join("");

  resultsEl.innerHTML = `
    <div class="panel">
      <h2>${title} ${colorChips(deck.color_label)}
        <span class="sub">· ${deck.total} cards</span></h2>
      ${deck.commander ? `<p class="result-note">Commander: <strong>${escapeHtml(deck.commander)}</strong></p>` : ""}
      ${deck.notes.map((n) => `<p class="result-note">${escapeHtml(n)}</p>`).join("")}
      ${missingNote(deck.missing, notFound)}
      <div class="deck-toolbar">
        <button type="button" id="copy-deck">Copy decklist</button>
        <button type="button" id="download-deck">Download .txt</button>
      </div>
      <div class="category-grid">${categories}</div>
      <p class="result-note">Click a card to see it; hover for why it was picked.</p>
    </div>`;

  $("copy-deck").addEventListener("click", async () => {
    await navigator.clipboard.writeText(deck.export_text);
    setStatus("Decklist copied to clipboard.", "ok");
  });
  $("download-deck").addEventListener("click", () => {
    const blob = new Blob([deck.export_text + "\n"], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "deck.txt";
    a.click();
    URL.revokeObjectURL(a.href);
  });
}

function renderIdeas(result, notFound) {
  const ideas = result.ideas;
  if (!ideas.length) {
    resultsEl.innerHTML = `<div class="panel"><h2>No decks found</h2>
      <p class="result-note">Not enough synergy in one color pair yet — try adding
      more cards, or analyze the collection to see what themes are close.</p>
      ${missingNote([], notFound)}</div>`;
    return;
  }
  resultsEl.innerHTML = `
    <div class="panel">
      <h2>Decks hiding in your collection
        <span class="sub">· ${result.format === "commander" ? "Commander" : "60-card"}</span></h2>
      ${missingNote(result.missing, notFound)}
      ${ideas.map((idea, i) => `
        <div class="idea">
          <span class="idea-title">${idea.commander ? `${escapeHtml(idea.commander)} — ` : ""}
            ${escapeHtml(idea.theme_name)} ${colorChips(idea.color_label)}</span>
          <span class="score">score ${idea.score}</span>
          <button type="button" data-idea="${i}">Build this</button>
          <span class="key-cards">Key cards: ${idea.key_cards.map(cardLink).join(", ")}</span>
        </div>`).join("")}
    </div>`;

  resultsEl.querySelectorAll("[data-idea]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idea = ideas[Number(btn.dataset.idea)];
      $("format").value = idea.commander ? "commander" : "60";
      $("format").dispatchEvent(new Event("change"));
      $("colors").value = idea.commander ? "" : idea.color_label;
      $("theme").value = idea.theme_key;
      $("commander").value = idea.commander ?? "";
      runAction("build");
    });
  });
}

function renderAnalysis(result, notFound) {
  const roles = result.roles.map((r) => `
    <tr><td>${escapeHtml(r.role.replace(/_/g, " "))}</td>
      <td class="num">${r.count}</td>
      <td class="examples">${r.examples.map(cardLink).join(", ")}</td></tr>`).join("");
  const maxScore = Math.max(...result.themes.map((t) => t.score), 1);
  const themes = result.themes.map((t) => `
    <div class="theme-row${t.viable ? " viable" : ""}">
      <span class="name">${escapeHtml(t.name)}</span>
      <span class="bar-track"><span class="bar" style="width:${(100 * t.score / maxScore).toFixed(0)}%"></span></span>
      <span class="score">${t.score}</span>
      <span class="theme-examples">${t.examples.map(cardLink).join(", ")}</span>
    </div>`).join("");

  resultsEl.innerHTML = `
    <div class="panel">
      <h2>Collection analysis
        <span class="sub">· ${result.total_cards} cards, ${result.unique_cards} unique</span></h2>
      ${missingNote(result.missing, notFound)}
      <h3>Functional roles</h3>
      <table class="role-table">${roles}</table>
      <h3>Synergy themes <span class="sub">(★ = enough on both sides to build around)</span></h3>
      ${themes}
    </div>`;
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

$("btn-build").addEventListener("click", () => runAction("build"));
$("btn-suggest").addEventListener("click", () => runAction("suggest"));
$("btn-analyze").addEventListener("click", () => runAction("analyze"));

function updateFormatVisibility() {
  const isCommander = $("format").value === "commander";
  $("commander-field").classList.toggle("hidden", !isCommander);
  $("colors").parentElement.classList.toggle("hidden", isCommander);
}
$("format").addEventListener("change", updateFormatVisibility);

$("file-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (file) {
    $("collection").value = await file.text();
    saveUiState();
  }
});

$("load-sample").addEventListener("click", async () => {
  const res = await fetch("examples/sample_collection.txt");
  $("collection").value = await res.text();
  saveUiState();
  setStatus("Sample collection loaded — try “Build deck”.", "ok");
});

// Persist edits: options immediately, the textarea lightly debounced.
let saveTimer = null;
$("collection").addEventListener("input", () => {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveUiState, 400);
});
for (const id of ["format", "colors", "theme", "commander", "playsets"]) {
  $(id).addEventListener("change", saveUiState);
}

restoreUiState();
boot();
