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
