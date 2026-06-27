  // Event data
  const eventsData = [{"title": "Tesla Owners at the British Motor Show 2026", "date": "21 Aug 2026", "time": "00:00", "startIso": "2026-08-21T00:00:00+01:00", "location": "Farnbrorough", "description": "Tesla Owners UK is delighted to have been invited to have a club stand at the British Motor Show at the Farnborough airport exhibition centre (where previous everything electric shows have been held).", "url": "https://ti.to/teslaownersuk/tesla-owners-at-the-british-motor-show-2026", "upcoming": true, "category": "exhibition", "region": "other", "banner": "https://do3z7e6uuakno.cloudfront.net/uploads/event/banner/1161029/13eed5f78da0deab010f6cbb7d817713.webp"}, {"title": "Everything Electric Greater London", "date": "11 Sep 2026", "time": "09:00", "startIso": "2026-09-11T09:00:00+01:00", "location": "Allianz Stadium", "description": "Everything Electric's last UK show of 2026 is at the new London venue of the Allianz Stadium. For our paid supporters, please log into your TOUK account after the summer to obtain your free ticket wor", "url": "https://ti.to/teslaownersuk/everything-electric-west-copy", "upcoming": true, "category": "exhibition", "region": "other", "banner": "https://do3z7e6uuakno.cloudfront.net/uploads/event/banner/1162351/0cf6ed09cca63238b266c1672fee021c.jpeg"}, {"title": "Tesla Gigafactory Berlin Tour 2026", "date": "18 Oct 2026", "time": "18:00", "startIso": "2026-10-18T18:00:00+01:00", "location": "Tesla Gigafactory Berlin-Brandenburg", "description": "Tesla has invited the official owners club of the United Kingdom to a private tour of their Gigafactory Berlin-Brandenburg for 2026. We meet in Berlin on Sunday 18 October for our club reception and d", "url": "https://ti.to/teslaownersuk/tesla-gigafactory-berlin-tour-2026-copy", "upcoming": true, "category": "meetup", "region": "other", "banner": "https://do3z7e6uuakno.cloudfront.net/uploads/event/banner/1162895/79667aa3fd4d76fb06cdd0f67e3a97d1.jpeg"}];

  // Dark mode toggle
  function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
  }

  // Load saved theme
  (function() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
  })();

  // Next upcoming start: from embedded eventsData so the counter rolls forward without redeploy.
  const MS_PER_DAY = 1000 * 60 * 60 * 24;

  function nextUpcomingStartIso() {
    const now = Date.now();
    let best = null;
    let bestT = Infinity;
    for (const ev of eventsData) {
      if (!ev.upcoming || !ev.startIso) continue;
      const t = new Date(ev.startIso).getTime();
      if (Number.isNaN(t) || t < now) continue;
      if (t < bestT) { bestT = t; best = ev.startIso; }
    }
    return best;
  }

  function updateDaysUntil() {
    const element = document.getElementById('daysUntilNext');
    if (!element) return;

    const nextIso = nextUpcomingStartIso();
    if (!nextIso) {
      element.textContent = '-';
      element.removeAttribute('data-next-event-date');
      return;
    }
    element.setAttribute('data-next-event-date', nextIso);

    const eventTime = new Date(nextIso).getTime();
    if (Number.isNaN(eventTime)) {
      element.textContent = '-';
      return;
    }

    const diffMs = eventTime - Date.now();
    const diffDays = Math.floor(diffMs / MS_PER_DAY);

    if (diffDays >= 0) {
      element.textContent = String(diffDays);
    } else {
      element.textContent = '-';
    }
  }

  updateDaysUntil();
  setInterval(updateDaysUntil, 60 * 1000);
  document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'visible') updateDaysUntil();
  });

  // Update "Last updated" timestamp dynamically
  function updateHealthTimestamp() {
    const element = document.getElementById('healthTimestamp');
    if (!element) return;

    const timestamp = element.getAttribute('data-timestamp');
    if (!timestamp) return;

    const updateTime = new Date(timestamp);
    const now = new Date();
    const diffSeconds = Math.floor((now - updateTime) / 1000);

    let timeAgo;
    if (diffSeconds < 60) {
      timeAgo = 'just now';
    } else if (diffSeconds < 3600) {
      const mins = Math.floor(diffSeconds / 60);
      timeAgo = `${mins} minute${mins > 1 ? 's' : ''} ago`;
    } else if (diffSeconds < 86400) {
      const hours = Math.floor(diffSeconds / 3600);
      timeAgo = `${hours} hour${hours > 1 ? 's' : ''} ago`;
    } else {
      const days = Math.floor(diffSeconds / 86400);
      timeAgo = `${days} day${days > 1 ? 's' : ''} ago`;
    }

    element.textContent = timeAgo;
  }

  // Update health timestamp on page load and every minute
  updateHealthTimestamp();
  setInterval(updateHealthTimestamp, 1000 * 60); // Update every minute

  // Filter state
  let activeCategory = 'all';
  let activeRegion = 'all';

  // Filter events
  function filterEvents() {
    const searchTerm = document.getElementById('searchBox').value.toLowerCase();
    const eventCards = document.querySelectorAll('.featured-event');
    let visibleCount = 0;

    eventCards.forEach((card, idx) => {
      const event = eventsData[idx];
      const matchesSearch = !searchTerm ||
        event.title.toLowerCase().includes(searchTerm) ||
        event.location.toLowerCase().includes(searchTerm) ||
        event.description.toLowerCase().includes(searchTerm);

      const matchesCategory = activeCategory === 'all' || card.dataset.category === activeCategory;
      const matchesRegion = activeRegion === 'all' || card.dataset.region === activeRegion;

      if (matchesSearch && matchesCategory && matchesRegion) {
        card.classList.remove('hidden');
        visibleCount++;
      } else {
        card.classList.add('hidden');
      }
    });

    // Update count
    const countSpan = document.getElementById('eventCount');
    if (searchTerm || activeCategory !== 'all' || activeRegion !== 'all') {
      countSpan.textContent = '(' + visibleCount + ' shown)';
    } else {
      countSpan.textContent = '';
    }
  }

  // Show event modal
  function showEventModal(idx) {
    const event = eventsData[idx];
    document.getElementById('modalTitle').textContent = event.title;
    document.getElementById('modalDate').textContent = event.date;
    document.getElementById('modalTime').textContent = event.time || 'TBA';
    document.getElementById('modalLocation').textContent = event.location || 'TBA';
    document.getElementById('modalDescription').textContent = event.description || 'No description available.';
    document.getElementById('modalLink').href = event.url || '#';

    if (!event.url) {
      document.getElementById('modalLink').style.display = 'none';
    } else {
      document.getElementById('modalLink').style.display = 'inline-block';
    }

    document.getElementById('eventModal').classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  // Close modal
  function closeModal() {
    document.getElementById('eventModal').classList.remove('active');
    document.body.style.overflow = '';
  }

  // Escape key closes modal
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeModal();
  });

  // Accordion toggle
  function toggleAccordion(id) {
    const button = document.getElementById('accordion-btn-' + id);
    const content = document.getElementById('accordion-' + id);
    const isExpanded = button.getAttribute('aria-expanded') === 'true';

    // Close all accordions
    document.querySelectorAll('.accordion-button').forEach(btn => {
      btn.setAttribute('aria-expanded', 'false');
    });
    document.querySelectorAll('.accordion-content').forEach(content => {
      content.classList.remove('active');
    });

    // Toggle current
    if (!isExpanded) {
      button.setAttribute('aria-expanded', 'true');
      content.classList.add('active');
    }
  }

  // Copy calendar URL to clipboard
  function copyCalendarURL() {
    const calendarURL = new URL('/tocuk.ics', window.location.href).href;
    navigator.clipboard.writeText(calendarURL).then(() => {
      const btn = event.currentTarget;
      const originalText = btn.textContent;
      btn.textContent = '✓ Copied!';
      btn.style.background = 'rgba(34, 197, 94, 0.2)';
      btn.style.borderColor = 'rgba(34, 197, 94, 0.4)';
      setTimeout(() => {
        btn.textContent = originalText;
        btn.style.background = '';
        btn.style.borderColor = '';
      }, 2000);
    }).catch(err => {
      console.error('Failed to copy:', err);
      alert('Failed to copy URL. Please copy manually: ' + calendarURL);
    });
  }

  // Filter by category
  function filterByCategory(category) {
    activeCategory = category;

    // Update chip styles
    document.querySelectorAll('.filter-chip[data-category]').forEach(chip => {
      if (chip.dataset.category === category) {
        chip.classList.add('active');
      } else {
        chip.classList.remove('active');
      }
    });

    filterEvents();
  }

  // Filter by region
  function filterByRegion(region) {
    activeRegion = region;

    // Update chip styles
    document.querySelectorAll('.filter-chip[data-region]').forEach(chip => {
      if (chip.dataset.region === region) {
        chip.classList.add('active');
      } else {
        chip.classList.remove('active');
      }
    });

    filterEvents();
  }

  // Device-specific calendar link handling
  (function() {
    var ua = navigator.userAgent || '';
    var isIOS = /iPhone|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    var isMac = /Macintosh|Mac OS X/.test(ua) && !isIOS;
    var isAndroid = /Android/.test(ua);

    document.querySelectorAll('a[href$=".ics"]').forEach(function(link) {
      var href = link.getAttribute('href');
      if (!href) return;
      var abs = new URL(href, window.location.href).href;

      if (isIOS || isMac) {
        // Use webcal:// protocol for Apple devices
        link.href = abs.replace(/^https?:\/\//, 'webcal://');
      } else if (isAndroid) {
        // Use Google Calendar render URL for Android
        link.href = 'https://calendar.google.com/calendar/render?cid=' + encodeURIComponent(abs);
      }
    });
  })();
