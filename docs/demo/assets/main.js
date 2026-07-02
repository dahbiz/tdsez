// TDSE-ⵣ — Responsive JavaScript for static site deployment
// Compatible with GitHub Pages (no framework required)

document.addEventListener('DOMContentLoaded', () => {
  // ──────────────────────────────────────────────
  // 1. THEME TOGGLE — persisted in localStorage
  // ──────────────────────────────────────────────
  const themeToggle = document.getElementById('themeToggle');
  const icon = themeToggle?.querySelector('.theme-icon');
  const stored = localStorage.getItem('tdse-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('tdse-theme', theme);
    if (icon) icon.textContent = theme === 'dark' ? '☀️' : '🌙';
  }

  if (stored) setTheme(stored);
  else if (prefersDark) setTheme('dark');
  else setTheme('light');

  themeToggle?.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    setTheme(current === 'dark' ? 'light' : 'dark');
  });

  // ──────────────────────────────────────────────
  // 2. MOBILE MENU — hamburger toggle with animation
  // ──────────────────────────────────────────────
  const menuToggle = document.getElementById('menuToggle');
  const mainNav = document.getElementById('mainNav');

  menuToggle?.addEventListener('click', () => {
    const isOpen = mainNav.classList.toggle('open');
    menuToggle.setAttribute('aria-expanded', String(isOpen));
    menuToggle.classList.toggle('active', isOpen);
  });

  // Close menu on outside click
  document.addEventListener('click', (e) => {
    if (mainNav?.classList.contains('open') &&
        !mainNav.contains(e.target) &&
        !menuToggle?.contains(e.target)) {
      mainNav.classList.remove('open');
      menuToggle?.setAttribute('aria-expanded', 'false');
      menuToggle?.classList.remove('active');
    }
  });

  // Close menu on link click (mobile)
  mainNav?.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      mainNav.classList.remove('open');
      menuToggle?.setAttribute('aria-expanded', 'false');
      menuToggle?.classList.remove('active');
    });
  });

  // Close menu on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && mainNav?.classList.contains('open')) {
      mainNav.classList.remove('open');
      menuToggle?.setAttribute('aria-expanded', 'false');
      menuToggle?.classList.remove('active');
      menuToggle?.focus();
    }
  });

  // ──────────────────────────────────────────────
  // 3. RESPONSIVE VIEWPORT HEIGHT — fixes mobile
  //    browser URL bar hiding/showing issues
  // ──────────────────────────────────────────────
  function setVH() {
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
  }
  setVH();
  window.addEventListener('resize', () => {
    // Debounce resize events
    clearTimeout(window._vhTimer);
    window._vhTimer = setTimeout(setVH, 100);
  });

  // ──────────────────────────────────────────────
  // 4. SCROLL-SPY — highlights active sidebar link
  // ──────────────────────────────────────────────
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.doc-sidebar a').forEach(link => {
    if (link.getAttribute('href')?.includes(currentPage)) {
      link.classList.add('active');
    }
  });

  // ──────────────────────────────────────────────
  // 5. INTERSECTION OBSERVER — animate elements
  //    on scroll into view (desktop only)
  // ──────────────────────────────────────────────
  const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

  if (!isTouchDevice) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    document.querySelectorAll('.feature-card, .link-card, .code-window, .highlight-grid').forEach(el => {
      el.classList.add('fade-in');
      observer.observe(el);
    });
  }

  // ──────────────────────────────────────────────
  // 6. COPY BUTTONS — clipboard with fallback
  // ──────────────────────────────────────────────
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const targetId = btn.getAttribute('data-target');
      const code = document.getElementById(targetId)?.textContent || '';
      try {
        await navigator.clipboard.writeText(code.trim());
        const original = btn.textContent;
        btn.textContent = 'Copied!';
        btn.style.background = 'var(--brand-1)';
        btn.style.color = 'white';
        setTimeout(() => {
          btn.textContent = original;
          btn.style.background = '';
          btn.style.color = '';
        }, 1500);
      } catch (err) {
        // Fallback: select and copy
        const textarea = document.createElement('textarea');
        textarea.value = code.trim();
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
          document.execCommand('copy');
          btn.textContent = 'Copied!';
          setTimeout(() => btn.textContent = 'Copy', 1500);
        } catch (e) {
          btn.textContent = 'Failed';
          setTimeout(() => btn.textContent = 'Copy', 1500);
        }
        document.body.removeChild(textarea);
      }
    });
  });

  // ──────────────────────────────────────────────
  // 7. FOOTER YEAR
  // ──────────────────────────────────────────────
  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  // ──────────────────────────────────────────────
  // 8. RESPONSIVE LAYOUT ADJUSTMENTS
  //    — Adjust hero visual on small screens
  //    — Lazy-load images below fold
  // ──────────────────────────────────────────────
  const heroVisual = document.querySelector('.hero-visual');
  if (heroVisual) {
    const heroWidth = heroVisual.offsetWidth;
    if (heroWidth < 400) {
      // On very small screens, hide the terminal visual
      heroVisual.style.display = 'none';
    }
  }

  // Lazy-load images below the fold
  if ('IntersectionObserver' in window) {
    const imgObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          if (img.dataset.src) {
            img.src = img.dataset.src;
            img.removeAttribute('data-src');
            imgObserver.unobserve(img);
          }
        }
      });
    }, { rootMargin: '200px' });

    document.querySelectorAll('img[data-src]').forEach(img => {
      imgObserver.observe(img);
    });
  }
});
