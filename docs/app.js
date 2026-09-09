const tissues = [
  { name: "Bladder", exact: 0.990, grounding: 0.910 },
  { name: "Ear", exact: 0.960, grounding: 0.910 },
  { name: "Eye", exact: 0.802, grounding: 0.693 },
  { name: "Heart", exact: 0.956, grounding: 0.899 },
  { name: "Ovary", exact: 0.950, grounding: 0.895 },
  { name: "Pancreas", exact: 0.937, grounding: 0.757 },
  { name: "Prostate", exact: 0.912, grounding: 0.759 },
  { name: "Salivary gland", exact: 0.760, grounding: 0.483 },
  { name: "Small intestine", exact: 0.719, grounding: 0.920 },
  { name: "Spleen", exact: 0.942, grounding: 0.923 },
];

const header = document.querySelector("[data-header]");
const menuButton = document.querySelector("[data-menu-button]");
const mobileNav = document.querySelector("[data-mobile-nav]");

function setMobileNavOpen(isOpen) {
  mobileNav.classList.toggle("is-open", isOpen);
  menuButton.setAttribute("aria-expanded", String(isOpen));
  menuButton.setAttribute("aria-label", isOpen ? "Close navigation" : "Open navigation");
  const icon = menuButton.querySelector("svg");
  if (icon) icon.outerHTML = `<i data-lucide="${isOpen ? "x" : "menu"}" aria-hidden="true"></i>`;
  window.lucide?.createIcons();
}

const updateHeader = () => header.classList.toggle("is-scrolled", window.scrollY > 24);
updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

menuButton.addEventListener("click", () => {
  setMobileNavOpen(!mobileNav.classList.contains("is-open"));
});

mobileNav.addEventListener("click", (event) => {
  if (!event.target.closest("a")) return;
  setMobileNavOpen(false);
});

const revealObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    entry.target.classList.add("is-visible");
    observer.unobserve(entry.target);
  });
}, { rootMargin: "0px 0px -8%", threshold: 0.08 });

document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

const chart = document.querySelector("[data-tissue-chart]");
const metricButtons = document.querySelectorAll("[data-metric]");

function renderTissueChart(metric) {
  chart.replaceChildren(...tissues.map((tissue) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-label" title="${tissue.name}">${tissue.name}</span>
      <span class="bar-track"><span class="bar-fill" style="--value: ${tissue[metric]}"></span></span>
      <span class="bar-value">${tissue[metric].toFixed(3)}</span>`;
    return row;
  }));
  chart.setAttribute("aria-label", `${metric === "exact" ? "Exact match" : "Evidence grounding"} by tissue`);
}

metricButtons.forEach((button) => {
  button.addEventListener("click", () => {
    metricButtons.forEach((item) => {
      const isActive = item === button;
      item.classList.toggle("is-active", isActive);
      item.setAttribute("aria-pressed", String(isActive));
    });
    renderTissueChart(button.dataset.metric);
  });
});
renderTissueChart("exact");

const lightbox = document.querySelector("[data-lightbox-dialog]");
const lightboxImage = lightbox.querySelector("[data-lightbox-image]");
const lightboxCaption = lightbox.querySelector("[data-lightbox-caption]");
const lightboxClose = lightbox.querySelector("[data-lightbox-close]");

document.querySelectorAll("[data-lightbox]").forEach((button) => {
  button.addEventListener("click", () => {
    lightboxImage.src = button.dataset.lightbox;
    lightboxImage.alt = button.querySelector("img").alt;
    lightboxCaption.textContent = button.dataset.caption;
    lightbox.showModal();
    document.body.classList.add("is-locked");
  });
});

function closeLightbox() {
  lightbox.close();
  lightboxImage.removeAttribute("src");
  document.body.classList.remove("is-locked");
}

lightboxClose.addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});
lightbox.addEventListener("close", () => document.body.classList.remove("is-locked"));

const copyButton = document.querySelector("[data-copy-citation]");
const copyStatus = document.querySelector("[data-copy-status]");
copyButton.addEventListener("click", async () => {
  const citation = document.querySelector("[data-citation]").textContent;
  try {
    await navigator.clipboard.writeText(citation);
    copyStatus.textContent = "BibTeX copied to clipboard.";
    copyButton.dataset.tooltip = "Copied";
  } catch {
    copyStatus.textContent = "Select the BibTeX text to copy it.";
  }
});

window.addEventListener("DOMContentLoaded", () => window.lucide?.createIcons());