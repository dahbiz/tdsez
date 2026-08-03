// TDSE-Z Docs — Custom JS for mobile nav, table wrappers, search
(function() {
  'use strict';

  // Mobile sidebar toggle
  const toggleBtn = document.querySelector('.mobile-nav-toggle');
  const sidebar = document.querySelector('.tdse-sidebar');
  const overlay = document.querySelector('.sidebar-overlay');

  if (toggleBtn && sidebar) {
    function openSidebar() {
      sidebar.classList.add('open');
      if (overlay) overlay.classList.add('open');
      document.body.style.overflow = 'hidden';
    }

    function closeSidebar() {
      sidebar.classList.remove('open');
      if (overlay) overlay.classList.remove('open');
      document.body.style.overflow = '';
    }

    toggleBtn.addEventListener('click', function() {
      if (sidebar.classList.contains('open')) {
        closeSidebar();
      } else {
        openSidebar();
      }
    });

    if (overlay) {
      overlay.addEventListener('click', closeSidebar);
    }

    // Close sidebar on link click
    sidebar.addEventListener('click', function(e) {
      if (e.target.tagName === 'A') {
        closeSidebar();
      }
    });
  }

  // Auto-wrap horizontal tables for mobile scrolling
  function wrapTables() {
    const tables = document.querySelectorAll('.rst-content table.docutils, .rst-content table.field-list');
    tables.forEach(function(table) {
      // Check if table is too wide for its container
      if (table.offsetWidth > table.parentElement.offsetWidth && table.parentElement.classList) {
        // Check if already wrapped
        if (!table.parentElement.classList.contains('table-wrapper')) {
          const wrapper = document.createElement('div');
          wrapper.className = 'table-wrapper';
          table.parentElement.insertBefore(wrapper, table);
          wrapper.appendChild(table);
        }
      }
    });
  }

  // Wrap tables on load and resize
  wrapTables();
  let resizeTimer;
  window.addEventListener('resize', function() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(wrapTables, 150);
  });

  // Smooth scroll for anchor links
  document.querySelectorAll('.rst-content a[href^="#"]').forEach(function(anchor) {
    anchor.addEventListener('click', function(e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Add copy button to code blocks
  function addCopyButtons() {
    document.querySelectorAll('pre').forEach(function(pre) {
      const btn = document.createElement('button');
      btn.className = 'code-copy-btn';
      btn.textContent = 'Copy';
      btn.style.cssText = 'position:absolute;top:0.5rem;right:0.5rem;background:rgba(255,255,255,0.9);border:1px solid #e2e8f0;border-radius:4px;padding:0.25rem 0.6rem;font-size:0.7rem;cursor:pointer;color:#475569;font-family:sans-serif;transition:all 0.15s;';
      btn.addEventListener('mouseenter', function() {
        this.style.background = 'rgba(14,165,233,0.9)';
        this.style.color = '#fff';
      });
      btn.addEventListener('mouseleave', function() {
        this.style.background = 'rgba(255,255,255,0.9)';
        this.style.color = '#475569';
      });
      btn.addEventListener('click', function() {
        const code = pre.textContent;
        navigator.clipboard.writeText(code).then(function() {
          btn.textContent = '✓';
          setTimeout(function() { btn.textContent = 'Copy'; }, 1500);
        }).catch(function() {
          btn.textContent = 'Failed';
          setTimeout(function() { btn.textContent = 'Copy'; }, 1500);
        });
      });
      pre.style.cssText = pre.style.cssText + 'position:relative;';
      pre.appendChild(btn);
    });
  }

  addCopyButtons();
})();
