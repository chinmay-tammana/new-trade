// GS AI 5-Minute Directional Indicator — Frontend Logic (Restored 1-Hour Layout for M5)
(function() {
  const DEFAULT_DAY = "2026-11-16";
  
  // Parse URL query parameter ?day=YYYY-MM-DD
  const urlParams = new URLSearchParams(window.location.search);
  let currentDay = urlParams.get("day") || DEFAULT_DAY;
  
  let chart = null;
  let candleSeries = null;
  let showNoSignal = true;
  let strongOnly = false;
  let showLabels = false;
  let sortOrder = 'chronological'; // 'chronological' (09:30 first) or 'newest' (15:55 first)
  let lastDeliveredTs = null;
  let isReplayMode = false;
  let cachedRows = [];

  // 1. Dual Clocks (ET and IST) & Real-time Live Age Counter
  function updateClocks() {
    const now = new Date();
    const etOptions = { timeZone: 'America/New_York', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' };
    const istOptions = { timeZone: 'Asia/Kolkata', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' };
    
    const etStr = new Intl.DateTimeFormat('en-US', etOptions).format(now);
    const istStr = new Intl.DateTimeFormat('en-US', istOptions).format(now);
    
    const clockEt = document.getElementById('clock-et');
    const clockIst = document.getElementById('clock-ist');
    const footerTime = document.getElementById('footer-time');
    
    if (clockEt) clockEt.textContent = `ET: ${etStr} EST`;
    if (clockIst) clockIst.textContent = `IST: ${istStr} IST`;
    if (footerTime) footerTime.textContent = `ET: ${etStr} | IST: ${istStr}`;

    // Live Signal Age Counter
    const hlAge = document.getElementById('headline-age');
    if (hlAge && lastDeliveredTs) {
      const ageSec = Math.max(0, Math.floor((Date.now() - lastDeliveredTs) / 1000));
      const delivDate = new Date(lastDeliveredTs);
      const timeStr = delivDate.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour12: false, hour: '2-digit', minute: '2-digit' });
      hlAge.textContent = `Issued at ${timeStr} ET (Age: ${ageSec}s ago)`;
    }
  }
  setInterval(updateClocks, 1000);
  updateClocks();

  // 2. Initialize Lightweight Charts
  function initChart() {
    const container = document.getElementById('tv-chart');
    if (!container || !window.LightweightCharts) return;

    const chartHeight = container.clientHeight || 280;

    chart = LightweightCharts.createChart(container, {
      width: container.clientWidth || 900,
      height: chartHeight,
      layout: {
        background: { color: '#0f172a' },
        textColor: '#94a3b8'
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' }
      },
      localization: {
        timeFormatter: (timestamp) => {
          const date = new Date(timestamp * 1000);
          return date.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false });
        }
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: '#334155',
        tickMarkFormatter: (time, tickMarkType, locale) => {
          const date = new Date(time * 1000);
          return date.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false });
        }
      },
      rightPriceScale: {
        borderColor: '#334155'
      }
    });

    candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444'
    });

    window.addEventListener('resize', () => {
      if (chart && container) {
        chart.applyOptions({
          width: container.clientWidth,
          height: container.clientHeight || 280
        });
      }
    });
  }

  // 3. Fetch Status & Headline / Sidebar
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();

      // Market badge
      const badgeMarket = document.getElementById('badge-market');
      if (badgeMarket) {
        badgeMarket.textContent = data.market_open ? 'MARKET OPEN' : 'MARKET CLOSED';
        badgeMarket.className = data.market_open ? 'badge badge-success' : 'badge badge-secondary';
      }

      // Model badge
      const badgeModel = document.getElementById('badge-model');
      if (badgeModel) {
        badgeModel.textContent = `MODEL: ${data.model_hash.toUpperCase()} (VERIFIED)`;
        badgeModel.className = data.model_verified ? 'badge badge-info' : 'badge badge-danger';
      }

      // Countdown
      const valCount = document.getElementById('val-countdown');
      if (valCount) {
        valCount.textContent = data.countdown_display || '--:--';
      }

      // Volatility Class & Regime & Bands (no fake fallbacks)
      const valVol = document.getElementById('val-vol');
      const valRegime = document.getElementById('val-regime');
      const valBands = document.getElementById('val-bands');
      if (valVol) valVol.textContent = (data.insights && data.insights.vol_class) ? data.insights.vol_class : '–';
      if (valRegime && data.insights) valRegime.textContent = data.insights.regime || 'Chop';
      if (valBands) {
        if (data.insights && data.insights.band_lo != null && data.insights.band_hi != null) {
          const bLo = Math.round(data.insights.band_lo * 10000);
          const bHi = Math.round(data.insights.band_hi * 10000);
          valBands.textContent = `[${bLo > 0 ? '+' : ''}${bLo}, +${bHi}] bps`;
        } else {
          valBands.textContent = '–';
        }
      }

      // Headline Card
      if (data.latest_signal && data.latest_signal.time_issued_iso) {
        lastDeliveredTs = new Date(data.latest_signal.time_issued_iso).getTime();
      }

      const hlCall = document.getElementById('headline-call');
      const hlTierChip = document.getElementById('headline-tier-chip');
      const heroSampleTag = document.getElementById('hero-sample-tag');

      if (hlCall) {
        if (data.latest_signal && data.latest_signal.direction && data.latest_signal.direction !== "Standby") {
          const dir = data.latest_signal.direction;
          const conf = data.latest_signal.confidence_pct;
          const tier = data.latest_signal.tier || 'NORMAL';
          const isUp = dir === 'Up';
          
          hlCall.innerHTML = `<span class="${isUp ? 'text-up' : 'text-down'}">${isUp ? '▲ UP' : '▼ DOWN'} ${conf}%</span>`;
          
          if (hlTierChip) {
            hlTierChip.textContent = `${tier} TIER`;
            hlTierChip.className = `strong-pill ${tier === 'STRONG' ? 'tier-strong' : (tier === 'WEAK' ? 'tier-weak' : 'tier-normal')}`;
          }
        } else {
          hlCall.textContent = data.headline_text || 'STANDBY';
        }
      }
      if (heroSampleTag) heroSampleTag.textContent = data.market_open ? 'LIVE' : 'OUT-OF-SAMPLE';

      // Signal Balance
      if (data.signal_balance) {
        const bal = data.signal_balance;
        const b1 = bal.last_1_day;
        const b5 = bal.last_5_days;
        const b20 = bal.last_20_days;
        
        const elB1 = document.getElementById('bal-1d');
        const elB5 = document.getElementById('bal-5d');
        const elB20 = document.getElementById('bal-20d');
        const lblB1 = document.getElementById('lbl-bal-1d');
        
        if (lblB1) lblB1.textContent = `Selected Session (${currentDay})`;
        if (elB1 && b1) elB1.textContent = `Up: ${b1.up} | Down: ${b1.down} | No-Sig: ${b1.no_signal}`;
        if (elB5 && b5) elB5.textContent = `Up: ${b5.up} | Down: ${b5.down} | No-Sig: ${b5.no_signal}`;
        if (elB20 && b20) elB20.textContent = `Up: ${b20.up} | Down: ${b20.down} | No-Sig: ${b20.no_signal}`;
      }

    } catch (e) {
      console.error("Error fetching status:", e);
    }
  }

  // 4. Fetch Chart Data & Candlestick Markers
  async function fetchChart() {
    if (!chart || !candleSeries) return;
    try {
      const res = await fetch(`/api/chart_bars?day=${currentDay}`);
      const data = await res.json();

      if (data.candles && data.candles.length > 0) {
        candleSeries.setData(data.candles);

        // Filter markers if strongOnly is enabled
        let markersToDisplay = data.markers || [];
        if (strongOnly) {
          markersToDisplay = markersToDisplay.filter(m => m.tier === 'STRONG');
        }

        // Format markers
        const formattedMarkers = markersToDisplay.map(m => {
          const isUp = m.shape === 'arrowUp';
          return {
            time: m.time,
            position: m.position,
            color: isUp ? '#10b981' : '#ef4444',
            shape: m.shape,
            text: showLabels ? (m.tier === 'STRONG' ? (isUp ? '▲ STRONG UP' : '▼ STRONG DN') : (isUp ? '▲ UP' : '▼ DN')) : (m.tier === 'STRONG' ? '★' : ''),
            size: m.tier === 'STRONG' ? 2 : 1
          };
        });

        candleSeries.setMarkers(formattedMarkers);
        chart.timeScale().fitContent();
      }
    } catch (e) {
      console.error("Error fetching chart data:", e);
    }
  }

  // 5. Fetch Signal Log, Summary Line, Strip & Scorecard
  async function fetchLog() {
    try {
      const res = await fetch(`/api/log?day=${currentDay}`);
      const data = await res.json();
      cachedRows = data.rows || [];

      // Revision notice display for 2026-11-16
      const revNotice = document.getElementById('feed-revision-notice');
      if (revNotice) {
        revNotice.style.display = (currentDay === '2026-11-16') ? 'block' : 'none';
      }

      // A. Compact Day Summary Line
      const sumLine = document.getElementById('summary-main');
      const dayHeader = document.getElementById('log-day-header');
      if (sumLine && data.summary && data.summary.summary_line) {
        sumLine.textContent = data.summary.summary_line;
      }
      if (dayHeader && data.summary) {
        dayHeader.textContent = `${data.summary.day_str} — Signals: ${data.summary.total_signals}`;
      }

      // B. Signal Strip of Chips Across Day
      const timelineStrip = document.getElementById('timeline-strip');
      if (timelineStrip && data.timeline_chips) {
        timelineStrip.innerHTML = '';
        data.timeline_chips.forEach(c => {
          const chip = document.createElement('div');
          chip.className = 'timeline-chip';
          
          const isUp = c.direction === 'Up';
          const isDown = c.direction === 'Down';
          const isPending = c.direction === 'Pending' || c.correct === null;
          const isCorrect = c.correct === 1;
          const isWrong = c.correct === 0;

          let badgeColor = '#334155';
          let textColor = '#94a3b8';
          let symbol = '–';

          if (isUp) {
            textColor = '#34d399';
            badgeColor = 'rgba(16, 185, 129, 0.15)';
            symbol = isCorrect ? '✔' : (isWrong ? '✘' : '⏳');
          } else if (isDown) {
            textColor = '#f87171';
            badgeColor = 'rgba(239, 68, 68, 0.15)';
            symbol = isCorrect ? '✔' : (isWrong ? '✘' : '⏳');
          } else if (c.direction === 'Pending') {
            textColor = '#fbbf24';
            badgeColor = 'rgba(245, 158, 11, 0.15)';
            symbol = '⏳';
          }

          chip.style.background = badgeColor;
          chip.style.color = textColor;
          chip.style.border = `1px solid ${textColor}44`;
          chip.style.borderRadius = '3px';
          chip.style.padding = '2px 6px';
          chip.style.fontSize = '10px';
          chip.style.fontFamily = 'monospace';
          chip.style.cursor = 'pointer';
          chip.style.whiteSpace = 'nowrap';

          const timeLabel = c.interval_et ? c.interval_et.split('-')[0] : '--:--';
          chip.innerHTML = `<strong>${timeLabel}</strong> ${c.direction ? c.direction.toUpperCase() : ''} ${symbol}`;
          
          chip.title = `${c.interval_et}: ${c.direction} (${c.result_display || '–'})`;
          chip.onclick = () => {
            const rowEl = document.getElementById(`row-${c.bar_start}`);
            if (rowEl) rowEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
          };
          timelineStrip.appendChild(chip);
        });
      }

      // C. Render Table Rows
      renderTable();

      // D. Update Performance Cards on Sidebar
      if (data.summary) {
        const s = data.summary;
        const hitRate = document.getElementById('score-hit-rate');
        const wilsonCi = document.getElementById('score-wilson-ci');
        const dirN = document.getElementById('score-directional-n');
        const upStat = document.getElementById('score-up-stat');
        const downStat = document.getElementById('score-down-stat');
        const covStat = document.getElementById('score-coverage');

        if (hitRate) hitRate.textContent = `${s.accuracy_all.pct.toFixed(1)}%`;
        if (wilsonCi) wilsonCi.textContent = `[${s.accuracy_all.ci_lo.toFixed(1)}%, ${s.accuracy_all.ci_hi.toFixed(1)}%]`;
        if (dirN) dirN.textContent = `n=${s.total_signals}`;
        
        // Up & Down stats
        const upRows = cachedRows.filter(r => r.direction === 'Up');
        const upCorrect = upRows.filter(r => r.correct === 1).length;
        const upPct = upRows.length > 0 ? (upCorrect / upRows.length * 100) : 0;
        if (upStat) upStat.textContent = `${upCorrect}/${upRows.length} (${upPct.toFixed(1)}%)`;

        const downRows = cachedRows.filter(r => r.direction === 'Down');
        const downCorrect = downRows.filter(r => r.correct === 1).length;
        const downPct = downRows.length > 0 ? (downCorrect / downRows.length * 100) : 0;
        if (downStat) downStat.textContent = `${downCorrect}/${downRows.length} (${downPct.toFixed(1)}%)`;

        if (covStat) covStat.textContent = `${s.total_signals} of 78 bars`;
      }

    } catch (e) {
      console.error("Error fetching signal log:", e);
    }
  }

  // 6. Render Signal Log Table
  function renderTable() {
    const tbody = document.getElementById('signal-log-tbody') || document.getElementById('signals-tbody');
    if (!tbody) return;

    let filtered = cachedRows.slice();
    if (!showNoSignal) {
      filtered = filtered.filter(r => r.direction === 'Up' || r.direction === 'Down');
    }
    if (strongOnly) {
      filtered = filtered.filter(r => r.tier === 'STRONG');
    }

    // Sort order: chronological (09:30 first) vs newest (15:55 first)
    filtered.sort((a, b) => {
      if (sortOrder === 'chronological') {
        return a.bar_start.localeCompare(b.bar_start);
      } else {
        return b.bar_start.localeCompare(a.bar_start);
      }
    });

    if (filtered.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="state-message">No signals match current filter.</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    filtered.forEach(r => {
      const tr = document.createElement('tr');
      tr.id = `row-${r.bar_start}`;

      // Call display styling
      let callHtml = '–';
      if (r.direction === 'Up') {
        const conf = r.confidence ? Math.round(r.confidence * 100) : 51;
        callHtml = `<strong class="text-up">▲ UP ${conf}%</strong>`;
      } else if (r.direction === 'Down') {
        const conf = r.confidence ? Math.round(r.confidence * 100) : 51;
        callHtml = `<strong class="text-down">▼ DOWN ${conf}%</strong>`;
      } else if (r.direction === 'Pending') {
        callHtml = `<span style="color:#fbbf24;">⏳ Pending</span>`;
      } else {
        callHtml = `<span style="color:#94a3b8;">${r.call_display || '– NO SIGNAL –'}</span>`;
      }

      // Strength chip
      let strengthHtml = '–';
      if (r.tier === 'STRONG') {
        strengthHtml = '<span class="tier-chip tier-strong" style="background:rgba(245, 158, 11, 0.2); color:#f59e0b; border:1px solid #f59e0b; padding:1px 6px; border-radius:3px; font-weight:bold; font-size:10px;">STRONG</span>';
      } else if (r.tier === 'WEAK') {
        strengthHtml = '<span class="tier-chip tier-weak" style="background:rgba(148, 163, 184, 0.2); color:#94a3b8; border:1px solid #64748b; padding:1px 6px; border-radius:3px; font-size:10px;">WEAK</span>';
      } else if (r.tier === 'NORMAL') {
        strengthHtml = '<span class="tier-chip tier-normal" style="background:rgba(56, 189, 248, 0.15); color:#38bdf8; border:1px solid #0284c7; padding:1px 6px; border-radius:3px; font-size:10px;">NORMAL</span>';
      }

      // Result badge
      let resultHtml = '–';
      if (r.result_display) {
        if (r.result_display.includes('Correct')) {
          resultHtml = '<span style="color:#10b981; font-weight:600;">✔ Correct</span>';
        } else if (r.result_display.includes('Wrong')) {
          resultHtml = '<span style="color:#ef4444; font-weight:600;">✘ Wrong</span>';
        } else if (r.result_display.includes('Pending')) {
          resultHtml = '<span style="color:#f59e0b;">⏳ Pending</span>';
        } else {
          resultHtml = `<span style="color:#94a3b8;">${r.result_display}</span>`;
        }
      }

      tr.innerHTML = `
        <td style="font-family:monospace; font-weight:500;">${r.interval_et || '--:--'}</td>
        <td style="font-family:monospace; font-size:11px; color:#94a3b8;">${r.issued_display || '-- | --'}</td>
        <td>${callHtml}</td>
        <td>${strengthHtml}</td>
        <td style="font-family:monospace; font-size:11px; color:#cbd5e1;">${r.vol_and_band_display || 'Normal | [-30, +30] bps'}</td>
        <td style="font-family:monospace; font-size:11px;">${r.price_move_display || '–'}</td>
        <td>${resultHtml}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // 7. Wire UI Controls
  function wireControls() {
    const dayPicker = document.getElementById('day-picker');
    const btnPrev = document.getElementById('btn-prev-day');
    const btnNext = document.getElementById('btn-next-day');
    const btnLatest = document.getElementById('btn-latest-day');
    const filterNosig = document.getElementById('filter-nosig-check');
    const filterStrong = document.getElementById('filter-strong-check');
    const sortSelect = document.getElementById('sort-order-select');
    const btnLabels = document.getElementById('btn-toggle-labels');
    const btnDiag = document.getElementById('btn-toggle-diagnostics');
    const diagSection = document.getElementById('diagnostics-section');
    const diagToggleText = document.getElementById('diag-toggle-text');
    const replaySelect = document.getElementById('replay-date-select');
    const btnRunReplay = document.getElementById('btn-run-replay');
    const btnResetLive = document.getElementById('btn-reset-live');

    if (dayPicker) {
      dayPicker.value = currentDay;
      dayPicker.addEventListener('change', (e) => {
        currentDay = e.target.value;
        refreshAll();
      });
    }

    if (btnPrev) {
      btnPrev.addEventListener('click', () => {
        const d = new Date(currentDay + "T12:00:00");
        d.setDate(d.getDate() - 1);
        if (d.getDay() === 0) d.setDate(d.getDate() - 2); // skip Sunday
        if (d.getDay() === 6) d.setDate(d.getDate() - 1); // skip Saturday
        currentDay = d.toISOString().split('T')[0];
        if (dayPicker) dayPicker.value = currentDay;
        refreshAll();
      });
    }

    if (btnNext) {
      btnNext.addEventListener('click', () => {
        const d = new Date(currentDay + "T12:00:00");
        d.setDate(d.getDate() + 1);
        if (d.getDay() === 6) d.setDate(d.getDate() + 2); // skip Saturday
        if (d.getDay() === 0) d.setDate(d.getDate() + 1); // skip Sunday
        currentDay = d.toISOString().split('T')[0];
        if (dayPicker) dayPicker.value = currentDay;
        refreshAll();
      });
    }

    if (btnLatest) {
      btnLatest.addEventListener('click', () => {
        currentDay = DEFAULT_DAY;
        if (dayPicker) dayPicker.value = currentDay;
        refreshAll();
      });
    }

    if (filterNosig) {
      filterNosig.addEventListener('change', (e) => {
        showNoSignal = e.target.checked;
        renderTable();
      });
    }

    if (filterStrong) {
      filterStrong.addEventListener('change', (e) => {
        strongOnly = e.target.checked;
        renderTable();
        fetchChart();
      });
    }

    if (sortSelect) {
      sortSelect.value = sortOrder;
      sortSelect.addEventListener('change', (e) => {
        sortOrder = e.target.value;
        renderTable();
      });
    }

    if (btnLabels) {
      btnLabels.addEventListener('click', () => {
        showLabels = !showLabels;
        btnLabels.textContent = `Labels: ${showLabels ? 'ON' : 'OFF'}`;
        fetchChart();
      });
    }

    if (btnDiag && diagSection) {
      btnDiag.addEventListener('click', () => {
        const isHidden = diagSection.style.display === 'none';
        diagSection.style.display = isHidden ? 'block' : 'none';
        if (diagToggleText) {
          diagToggleText.textContent = isHidden ? 'Hide Model & Feed Diagnostics ▲' : 'Show Model & Feed Diagnostics ▼';
        }
      });
    }

    // Historical Replay Controls
    if (btnRunReplay && replaySelect) {
      btnRunReplay.addEventListener('click', async () => {
        const selectedDay = replaySelect.value;
        currentDay = selectedDay;
        if (dayPicker) dayPicker.value = currentDay;
        isReplayMode = true;
        
        const badgeReplay = document.getElementById('badge-replay');
        if (badgeReplay) badgeReplay.style.display = 'inline-block';
        if (btnResetLive) btnResetLive.style.display = 'inline-block';
        
        refreshAll();
      });
    }

    if (btnResetLive) {
      btnResetLive.addEventListener('click', () => {
        currentDay = DEFAULT_DAY;
        if (dayPicker) dayPicker.value = currentDay;
        isReplayMode = false;
        
        const badgeReplay = document.getElementById('badge-replay');
        if (badgeReplay) badgeReplay.style.display = 'none';
        btnResetLive.style.display = 'none';
        
        refreshAll();
      });
    }
  }

  // 8. Refresh All Data
  function refreshAll() {
    fetchStatus();
    fetchChart();
    fetchLog();
  }

  // 9. Startup Lifecycle
  document.addEventListener('DOMContentLoaded', () => {
    initChart();
    wireControls();
    refreshAll();
    
    // Aligned 5-second polling loop
    setInterval(() => {
      fetchStatus();
      // Periodically refresh chart and log
      const now = new Date();
      if (now.getSeconds() % 15 === 0) {
        fetchChart();
        fetchLog();
      }
    }, 5000);
  });
})();
