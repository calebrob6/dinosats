// Step-by-step help card with a spotlight on the part of the UI it describes (after DeltaBit / TerraBit).
// Each step is {label, title, body, target?, placement?}; body is an HTML string or a function returning one, and
// target a function returning the element to spotlight (no target: a centered card over the dimmed page).

const NARROW = 640;

export function createTour({ steps, trigger, storageKey }) {
  const overlay = document.createElement("div");
  overlay.className = "tut-overlay";
  const card = document.createElement("div");
  card.className = "tut-card";
  card.tabIndex = -1;
  card.setAttribute("role", "dialog");
  card.setAttribute("aria-label", "Help");
  card.innerHTML = `
    <button type="button" class="tut-close" aria-label="Close">×</button>
    <div class="tut-step-label"></div>
    <div class="tut-title"></div>
    <div class="tut-body"></div>
    <div class="tut-nav">
      <div class="tut-dots"></div>
      <button type="button" class="tut-btn ghost" data-action="back">Back</button>
      <button type="button" class="tut-btn primary" data-action="next">Next →</button>
    </div>
    <label class="tut-optout"><input type="checkbox" /> Don't show this on startup</label>`;
  document.body.append(overlay, card);

  const $ = (sel) => card.querySelector(sel);
  const optOut = $(".tut-optout input");
  const hidden = () => {
    try {
      return localStorage.getItem(storageKey) === "1";
    } catch {
      return false;
    }
  };
  optOut.checked = hidden();
  optOut.addEventListener("change", () => {
    try {
      if (optOut.checked) localStorage.setItem(storageKey, "1");
      else localStorage.removeItem(storageKey);
    } catch {
      // storage unavailable (e.g. blocked cookies): the tour just shows again next time
    }
  });

  let active = false;
  let step = 0;
  let returnFocus = null;

  const targetRect = (s) => {
    const r = s.target?.()?.getBoundingClientRect();
    return r && r.width && r.height ? r : null;
  };

  function place() {
    const s = steps[step];
    const r = targetRect(s);
    const pad = s.padding ?? 8;
    overlay.classList.toggle("has-spotlight", !!r);
    if (r) {
      overlay.style.setProperty("--tut-spot-x", `${r.left - pad}px`);
      overlay.style.setProperty("--tut-spot-y", `${r.top - pad}px`);
      overlay.style.setProperty("--tut-spot-w", `${r.width + 2 * pad}px`);
      overlay.style.setProperty("--tut-spot-h", `${r.height + 2 * pad}px`);
    }
    const [vw, vh, margin, gap] = [innerWidth, innerHeight, 12, 16];
    const narrow = vw <= NARROW;
    card.style.width = narrow ? `${vw - 2 * margin}px` : "";
    const { width, height } = card.getBoundingClientRect();
    let left = (vw - width) / 2;
    let top = (vh - height) / 2;
    if (r && narrow) {
      // Full-width card at the bottom, or at the top when the spotlight is in the lower half.
      left = margin;
      top = r.top + r.height / 2 > vh / 2 ? margin : vh - height - margin;
    } else if (r) {
      const fits = (x, y) => x >= margin && y >= margin && x + width <= vw - margin && y + height <= vh - margin;
      const at = {
        right: [r.right + gap, r.top + (r.height - height) / 2],
        left: [r.left - width - gap, r.top + (r.height - height) / 2],
        bottom: [r.left + (r.width - width) / 2, r.bottom + gap],
        top: [r.left + (r.width - width) / 2, r.top - height - gap],
      };
      const order = [s.placement ?? "right", "left", "right", "bottom", "top"];
      [left, top] = at[order.find((p) => fits(at[p][0], Math.max(margin, Math.min(at[p][1], vh - height - margin))))] ?? [left, top];
    }
    card.style.left = `${Math.max(margin, Math.min(left, vw - width - margin))}px`;
    card.style.top = `${Math.max(margin, Math.min(top, vh - height - margin))}px`;
  }

  function render() {
    const s = steps[step];
    $(".tut-step-label").textContent = s.label ?? `Step ${step} / ${steps.length - 1}`;
    $(".tut-title").textContent = s.title;
    $(".tut-body").innerHTML = typeof s.body === "function" ? s.body() : s.body;
    $('[data-action="back"]').disabled = step === 0;
    $('[data-action="next"]').textContent = step === steps.length - 1 ? "Get started" : "Next →";
    $(".tut-dots").replaceChildren(
      ...steps.map((_, i) => {
        const dot = document.createElement("span");
        dot.className = `tut-dot${i === step ? " is-active" : ""}`;
        return dot;
      }),
    );
    requestAnimationFrame(place);
  }

  function go(i) {
    if (i < 0 || i >= steps.length) return;
    step = i;
    render();
  }

  function start() {
    if (active) return;
    active = true;
    step = 0;
    overlay.classList.add("is-active");
    card.classList.add("is-active");
    trigger?.setAttribute("aria-expanded", "true");
    render();
    // Take the keyboard (Esc, arrows) unless the user is already typing a query.
    const focused = document.activeElement;
    if (!focused?.value) {
      returnFocus = focused;
      card.focus({ preventScroll: true });
    }
    addEventListener("resize", place);
  }

  function stop() {
    if (!active) return;
    active = false;
    overlay.classList.remove("is-active", "has-spotlight");
    card.classList.remove("is-active");
    trigger?.setAttribute("aria-expanded", "false");
    removeEventListener("resize", place);
    if (card.contains(document.activeElement)) (returnFocus ?? document.body).focus?.({ preventScroll: true });
    returnFocus = null;
  }

  card.addEventListener("click", (e) => {
    const action = e.target.closest("[data-action]")?.dataset.action;
    if (action === "back") go(step - 1);
    else if (action === "next") step === steps.length - 1 ? stop() : go(step + 1);
  });
  $(".tut-close").addEventListener("click", stop);
  trigger?.addEventListener("click", () => (active ? stop() : start()));

  document.addEventListener("keydown", (e) => {
    if (!active || e.ctrlKey || e.metaKey || e.altKey) return;
    if (!card.contains(e.target) && e.target.closest?.("input, textarea, select")) return;
    if (e.key === "Escape") {
      e.preventDefault();
      stop();
    } else if (e.key === "ArrowRight" || (e.key === "Enter" && !e.target.closest?.("button"))) {
      e.preventDefault();
      step === steps.length - 1 ? stop() : go(step + 1);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      go(step - 1);
    }
  });

  return {
    start,
    stop,
    get active() {
      return active;
    },
    /** Opens the tour unless "Don't show this on startup" was checked on an earlier visit. */
    autoStart() {
      if (!hidden()) start();
    },
  };
}
