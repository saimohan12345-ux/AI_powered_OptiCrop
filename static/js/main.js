/**
 * OptiCrop — Main JavaScript
 * Handles: AOS init · Dark mode toggle · Navbar scroll · Global utilities
 */

'use strict';

// ─── AOS Scroll Animations ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  if (typeof AOS !== 'undefined') {
    AOS.init({
      duration: 700,
      easing:   'ease-out-cubic',
      once:     true,
      offset:   60,
    });
  }

  initThemeToggle();
  initNavbarScroll();
  initSliderSync();
});

// ─── Theme Toggle ──────────────────────────────────────────────────────────
function initThemeToggle() {
  const btn  = document.getElementById('theme-toggle');
  const icon = document.getElementById('theme-icon');
  if (!btn) return;

  // Load saved theme
  const saved = localStorage.getItem('oc-theme') || 'dark';
  applyTheme(saved);

  btn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next    = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    localStorage.setItem('oc-theme', next);
  });

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    if (icon) {
      icon.className = theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
    }
  }
}

// ─── Navbar Scroll Effect ─────────────────────────────────────────────────
function initNavbarScroll() {
  const navbar = document.getElementById('main-navbar');
  if (!navbar) return;

  const handler = () => {
    if (window.scrollY > 40) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  };

  window.addEventListener('scroll', handler, { passive: true });
  handler(); // run once on load
}

// ─── Global Slider ↔ Number Sync ─────────────────────────────────────────
// This runs globally in case the prediction page is loaded.
// The prediction.html also has its own scoped version, but this is a safety fallback.
function initSliderSync() {
  document.querySelectorAll('.param-slider').forEach(slider => {
    if (slider.dataset.bound) return; // avoid double binding
    slider.dataset.bound = '1';

    const numId = slider.dataset.target;
    const num   = numId ? document.getElementById(numId) : null;
    if (!num) return;

    slider.addEventListener('input', () => {
      num.value = slider.value;
    });

    num.addEventListener('input', () => {
      let v   = parseFloat(num.value);
      const min = parseFloat(slider.min);
      const max = parseFloat(slider.max);
      if (!isNaN(v)) {
        v = Math.max(min, Math.min(max, v));
        slider.value = v;
      }
    });
  });
}

// ─── Utility: format number ────────────────────────────────────────────────
window.OC = {
  /**
   * Format a number with fixed decimal places.
   * @param {number} n
   * @param {number} decimals
   * @returns {string}
   */
  fmt(n, decimals = 1) {
    if (n === null || n === undefined) return '—';
    return Number(n).toFixed(decimals);
  },

  /**
   * Debounce a function.
   * @param {Function} fn
   * @param {number} delay
   * @returns {Function}
   */
  debounce(fn, delay = 300) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), delay);
    };
  },

  /**
   * Animate a number from 0 → target.
   * @param {HTMLElement} el
   * @param {number} target
   * @param {string} suffix
   * @param {number} duration ms
   */
  animateCount(el, target, suffix = '', duration = 1200) {
    const isFloat = !Number.isInteger(target);
    const start   = performance.now();
    const step = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const e = 1 - Math.pow(1 - t, 3); // ease-out-cubic
      const v = target * e;
      el.textContent = (isFloat ? v.toFixed(1) : Math.round(v)) + suffix;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  },
};
