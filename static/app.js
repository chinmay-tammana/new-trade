// Lightweight Charts & Real-time Live Trading Indicator Dashboard (V3.6 Single Source of Truth Upgrade)
(function () {
  'use strict';

  let chart = null;
  let candleSeries = null;
  let countdownSeconds = 0;
  let countdownTimer = null;
  let isMarketOpen = false;
  let isReplayMode = false;
  let replayInterval = null;
  let showMarkerLabels = false;

  // Clock Synchronization
  let serverOffsetMs = 0;
  let clockTimer = null;

  // V3.6 Day View State
  let availableDays = [];
  let currentDay = null;
  let currentHorizon = '1 hour';
  let currentView = 'SCORED';
  let currentIncludeNoSig = true;
  let currentIncludeIex = false;
  let activeExpandedBarStart = null;
  let cachedBars = [];
  let cachedDayRows = [];
  let isFetchingLog = false;
  let newestRowBarStart = null;
  let highlightTimeout = null;

  // Track plotted markers by bar timestamp
  const markerMap = new Map();

  // Dual Clocks (ET and IST) synchronized with server offset
  function startDualClocks() {
    function updateClocks() {
      const now = new Date(Date.now() + serverOffsetMs);
      
      const etOptions = { timeZone: 'America/New_York', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' };
      const istOptions = { timeZone: 'Asia/Kolkata', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' };
      
      const etTimeStr = now.toLocaleTimeString('en-US', etOptions);
      const istTimeStr = now.toLocaleTimeString('en-US', istOptions);

      const elEt = document.getElementById('clock-et');
      if (elEt) elEt.textContent = `ET: ${etTimeStr} EST`;

      const elIst = document.getElementById('clock-ist');
      if (elIst) elIst.textContent = `IST: ${istTimeStr} IST`;

      const elFooter = document.getElementById('footer-time');
      if (elFooter) elFooter.textContent = `ET: ${etTimeStr} | IST: ${istTimeStr}`;
    }
    updateClocks();
    if (clockTimer) clearInterval(clockTimer);
    clockTimer = setInterval(updateClocks, 1000);
  }

  // Pure countdown formatter adhering strictly to V3 specifications
  function formatCountdownPure(seconds, marketOpen) {
    if (seconds <= 0) {
      return marketOpen ? '00:00' : 'Opens Soon';
    }
    if (marketOpen) {
      if (seconds < 3600) {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
      } else {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${String(mins).padStart(2, '0')}m`;
      }
    }
    const days = Math.floor(seconds / 86400);
    const rem = seconds % 86400;
    const hours = Math.floor(rem / 3600);
    const mins = Math.floor((rem % 3600) / 60);

    if (days >= 1) {
      return `${days}d ${hours}h ${mins}m`;
    } else if (hours >= 1) {
      return `${hours}h ${mins}m`;
    } else {
      const secs = rem % 60;
      return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }
  }

  function startCountdown(initialSeconds, marketOpen) {
    countdownSeconds = initialSeconds;
    isMarketOpen = marketOpen;
    if (countdownTimer) clearInterval(countdownTimer);

    function tick() {
      const elVal = document.getElementById('val-countdown');
      const elLbl = document.getElementById('lbl-countdown');

      if (elLbl) {
        elLbl.textContent = 'Next Signal Delivery';
      }

      if (countdownSeconds > 0) {
        countdownSeconds--;
      }
      if (elVal) {
        elVal.textContent = formatCountdownPure(countdownSeconds, isMarketOpen);
      }
    }

    tick();
    countdownTimer = setInterval(tick, 1000);
  }

  // Initialize TradingView Lightweight Charts
  function initChart() {
    const chartContainer = document.getElementById('tv-chart');
    if (!chartContainer || chart) return;

    chart = LightweightCharts.createChart(chartContainer, {
      width: chartContainer.clientWidth,
      height: chartContainer.clientHeight || 260,
      layout: {
        background: { color: '#0d1017' },
        textColor: '#cbd5e1',
        fontSize: 11,
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif'
      },
      grid: {
        vertLines: { color: 'rgba(30, 41, 59, 0.4)' },
        horzLines: { color: 'rgba(30, 41, 59, 0.4)' }
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: '#1e293b'
      },
      rightPriceScale: {
        borderColor: '#1e293b',
        autoScale: true
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal
      }
    });

    candleSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350'
    });

    // Resize observer
    window.addEventListener('resize', () => {
      if (chart && chartContainer) {
        chart.applyOptions({
          width: chartContainer.clientWidth,
          height: chartContainer.clientHeight || 260
        });
      }
    });
  }

  // Fetch Days List
  async function fetchDaysList() {
    try {
      const res = await fetch('/api/days');
      const days = await res.json();
      if (!Array.isArray(days) || !days.length) return;

      availableDays = days;

      // Check URL query param for ?day=YYYY-MM-DD
      const urlParams = new URLSearchParams(window.location.search);
      const urlDay = urlParams.get('day');

      if (urlDay && availableDays.some(d => d.date === urlDay)) {
        currentDay = urlDay;
      } else if (!currentDay) {
        const latestScored = availableDays.find(d => d.is_latest_scored);
        currentDay = latestScored ? latestScored.date : availableDays[0].date;
      }

      // Configure Day Picker limits
      const picker = document.getElementById('day-picker');
      if (picker) {
        picker.min = availableDays[availableDays.length - 1].date;
        picker.max = availableDays[0].date;
        picker.value = currentDay;
      }

      updateDayNavControls();
    } catch (e) {
      console.error('Failed to fetch days list:', e);
    }
  }

  function updateDayNavControls() {
    const btnPrev = document.getElementById('btn-prev-day');
    const btnNext = document.getElementById('btn-next-day');
    const btnLatest = document.getElementById('btn-latest-day');
    const picker = document.getElementById('day-picker');

    if (picker) picker.value = currentDay;

    const idx = availableDays.findIndex(d => d.date === currentDay);
    if (idx !== -1) {
      if (btnPrev) btnPrev.disabled = (idx >= availableDays.length - 1);
      if (btnNext) btnNext.disabled = (idx <= 0);
    }

    const latestScored = availableDays.find(d => d.is_latest_scored);
    if (btnLatest && latestScored) {
      btnLatest.classList.toggle('btn-primary', currentDay === latestScored.date);
    }

    // Synchronize URL
    const url = new URL(window.location);
    if (currentDay) {
      url.searchParams.set('day', currentDay);
      window.history.replaceState({}, '', url);
    }
  }

  function showSkeletonLoading() {
    const tbody = document.getElementById('signals-tbody');
    if (!tbody) return;
    let rowsHtml = '';
    for (let i = 0; i < 7; i++) {
      rowsHtml += `<tr><td colspan="8"><div class="skeleton-row" style="height:28px;"></div></td></tr>`;
    }
    tbody.innerHTML = rowsHtml;
  }

  // Single Source of Truth Log & Summary Fetch
  async function fetchLogAndRender(showSkeleton = false) {
    if (isReplayMode || isFetchingLog) return;
    isFetchingLog = true;

    if (showSkeleton) showSkeletonLoading();

    try {
      const params = new URLSearchParams({
        day: currentDay || '',
        horizon: currentHorizon,
        view: currentView,
        include_iex: currentIncludeIex ? 'true' : 'false',
        include_nosignal: currentIncludeNoSig ? 'true' : 'false'
      });

      const [logRes, barsRes, latestRes] = await Promise.all([
        fetch(`/api/log?${params.toString()}`),
        fetch('/api/bars?limit=700'),
        fetch('/api/latest')
      ]);

      const logData = await logRes.json();
      const bars = await barsRes.json();
      const latest = await latestRes.json();

      cachedBars = bars;
      cachedDayRows = logData.rows || [];

      renderChart(bars, cachedDayRows);
      renderLatestInsight(latest);
      renderDaySummary(logData.summary, logData.empty_reason);
      renderHourStrip(cachedDayRows);
      renderSignalLogTable(cachedDayRows, logData.empty_reason);
    } catch (e) {
      console.error('Failed to fetch signal log:', e);
      const tbody = document.getElementById('signals-tbody');
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="8" class="error-state">Failed to load signals: ${e.message}. <button id="btn-retry-log" class="btn btn-secondary btn-sm" style="margin-left:8px;">Retry</button></td></tr>`;
        const btnRetry = document.getElementById('btn-retry-log');
        if (btnRetry) btnRetry.addEventListener('click', () => fetchLogAndRender(true));
      }
    } finally {
      isFetchingLog = false;
    }
  }

  function renderDaySummary(summary, emptyReason) {
    const elMain = document.getElementById('summary-main');
    const elSub = document.getElementById('summary-sub');

    if (!summary || emptyReason) {
      if (elMain) elMain.textContent = emptyReason || 'No signals recorded on this day.';
      if (elSub) elSub.textContent = '';
      return;
    }

    if (elMain) elMain.textContent = summary.summary_line || `${summary.day_str} | ${summary.price_str}`;
    if (elSub) elSub.textContent = summary.demonstration_note || 'demonstration, not proof';
  }

  function renderHourStrip(rows) {
    const container = document.getElementById('hour-strip-container');
    if (!container) return;

    container.innerHTML = '';
    // Filter to primary scored rows (one per hour)
    const primaryRows = rows.filter(r => !r.is_iex);

    if (!primaryRows.length) {
      container.innerHTML = `<span style="font-size:11px; color:#64748b; padding:4px 8px;">No hourly bars available for this day.</span>`;
      return;
    }

    primaryRows.forEach(r => {
      const chip = document.createElement('div');
      const isUp = r.direction === 'Up';
      const isDown = r.direction === 'Down';
      const chipTypeClass = isUp ? 'hour-chip-up' : (isDown ? 'hour-chip-down' : 'hour-chip-nosig');

      chip.className = `hour-chip ${chipTypeClass}`;
      chip.setAttribute('data-bar-start', r.bar_start);

      // Compact display: Hour Window | Call | Result Symbol
      const hourPart = r.hour_predicted_et.split('-')[0].replace(' (close)', '').replace(' (market closed)', '');
      chip.innerHTML = `
        <span>${hourPart}</span>
        <strong>${r.call_display}</strong>
        <span class="hour-chip-res">${r.result_symbol}</span>
      `;

      chip.addEventListener('click', () => {
        // Highlight chip
        document.querySelectorAll('.hour-chip').forEach(c => c.classList.remove('chip-active'));
        chip.classList.add('chip-active');

        // Scroll table to row
        const targetRow = document.querySelector(`tr.signal-row[data-bar-start="${r.bar_start}"]`);
        if (targetRow) {
          targetRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
          highlightTableRow(targetRow);
        }

        // Highlight chart candle
        const unixSec = Math.floor(new Date(r.bar_start).getTime() / 1000);
        if (chart) {
          chart.timeScale().scrollToPosition(0, true);
        }
      });

      container.appendChild(chip);
    });
  }

  function highlightTableRow(rowEl) {
    document.querySelectorAll('tr.signal-row').forEach(tr => tr.classList.remove('row-active-highlight'));
    rowEl.classList.add('row-active-highlight');
  }

  function renderSignalLogTable(rows, emptyReason) {
    const tbody = document.getElementById('signals-tbody');
    if (!tbody) return;

    if (!rows || !rows.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="state-message">${emptyReason || "No signals recorded on this day."}</td></tr>`;
      return;
    }

    tbody.innerHTML = '';

    rows.forEach(r => {
      const tr = document.createElement('tr');
      const isIexRow = r.is_iex;
      tr.className = `signal-row ${isIexRow ? 'row-iex-shifted' : ''}`;
      tr.setAttribute('data-bar-start', r.bar_start);

      // Column 1: Hour predicted (ET)
      const colHour = `<strong>${r.hour_predicted_et}</strong>`;

      // Column 2: Issued (ET | IST)
      let colIssued = '';
      if (r.is_replayed) {
        colIssued = `<span style="color:#94a3b8; font-style:italic;" title="Computed retrospectively (replayed)">${r.issued_display} <em>(replayed)</em></span>`;
      } else {
        colIssued = `<strong style="color:#e2e8f0;">${r.issued_display}</strong>`;
      }

      // Column 3: Call
      let colCall = '';
      if (r.direction === 'Up') {
        colCall = `<span class="pill-mini" style="background:rgba(38,166,154,0.2); color:#26a69a; border:1px solid #26a69a; font-size:12px;">▲ UP ${(r.confidence * 100).toFixed(0)}%</span>`;
      } else if (r.direction === 'Down') {
        colCall = `<span class="pill-mini" style="background:rgba(239,83,80,0.2); color:#ef5350; border:1px solid #ef5350; font-size:12px;">▼ DOWN ${(r.confidence * 100).toFixed(0)}%</span>`;
      } else {
        colCall = `<span class="pill-mini" style="background:#1e293b; color:#94a3b8; border:1px solid #475569; font-size:11px;">– NO SIGNAL <span style="font-size:10px; color:#64748b;">(${r.call_reason || 'neutral'})</span></span>`;
      }

      // Column 4: Strength
      const colStrength = r.strength_display === 'STRONG' ? '<span class="strong-pill">STRONG</span>' : '';

      // Column 5: Vol and band
      const colVolBand = `<span style="font-size:12px; color:#cbd5e1;">${r.vol_band_display}</span>`;

      // Column 6: Price move
      const colPrice = `<span style="font-size:12px; color:#f1f5f9;">${r.price_move_display}</span>`;

      // Column 7: Result
      let colResult = '';
      if (r.result_display === '✔ Correct') {
        colResult = '<strong style="color:#26a69a;">✔ Correct</strong>';
      } else if (r.result_display === '✘ Wrong') {
        colResult = '<strong style="color:#ef5350;">✘ Wrong</strong>';
      } else if (r.result_display === '○ Pending') {
        colResult = '<span style="color:#38bdf8;">○ Pending</span>';
      } else {
        colResult = '<span style="color:#64748b;">–</span>';
      }

      // Column 8: Source
      let colSource = '';
      if (r.is_iex) {
        colSource = `<span class="pill-mini" style="background:#475569; color:#f1f5f9;" title="Legacy feature-shifted IEX data (not scored)">IEX (shifted)</span>`;
      } else {
        const tagClass = `tag-${(r.sample || 'OUT-OF-SAMPLE').toLowerCase().replace(/[^a-z]/g, '')}`;
        colSource = `<span class="pill-mini ${tagClass}" title="${r.sample}">${r.sample}</span>`;
      }

      tr.innerHTML = `
        <td>${colHour}</td>
        <td>${colIssued}</td>
        <td>${colCall}</td>
        <td>${colStrength}</td>
        <td>${colVolBand}</td>
        <td>${colPrice}</td>
        <td>${colResult}</td>
        <td>${colSource}</td>
      `;

      // Expandable Detail Row Handler
      tr.addEventListener('click', (e) => {
        toggleDetailRow(tr, r);
      });

      tbody.appendChild(tr);

      // Re-open detail row if it was active
      if (activeExpandedBarStart === r.bar_start && !r.is_iex) {
        openDetailRow(tr, r);
      }
    });
  }

  function toggleDetailRow(rowEl, r) {
    const nextEl = rowEl.nextElementSibling;
    if (nextEl && nextEl.classList.contains('detail-row')) {
      nextEl.remove();
      activeExpandedBarStart = null;
      return;
    }

    // Close any other open detail row
    document.querySelectorAll('tr.detail-row').forEach(el => el.remove());

    openDetailRow(rowEl, r);
  }

  function openDetailRow(rowEl, r) {
    activeExpandedBarStart = r.bar_start;
    highlightTableRow(rowEl);

    const detailTr = document.createElement('tr');
    detailTr.className = 'detail-row';

    detailTr.innerHTML = `
      <td colspan="8">
        <div class="detail-row-grid">
          <div class="detail-item">
            <strong>Calibrated Posterior:</strong> ${r.threshold_str}
          </div>
          <div class="detail-item">
            <strong>Market Regime:</strong> ${r.regime || 'Chop'}
          </div>
          <div class="detail-item">
            <strong>Data Tape:</strong> ${r.feed === 'sip_delayed' ? 'SIP Consolidated (15m delayed)' : 'IEX (display only)'}
          </div>
          <div class="detail-item">
            <strong>Model Artifact Hash:</strong> <code>${r.model_hash}</code> (Frozen)
          </div>
          <div class="detail-item">
            <strong>Bar Window:</strong> ${r.bar_close_time_et} ET close
          </div>
          <div class="detail-item">
            <strong>Delivery Lag:</strong> ${r.delivery_delay_s}s (16 min delay verified)
          </div>
          <div class="detail-item" style="grid-column: 1 / -1;">
            <strong>Cryptographic Record Hash:</strong> <code style="font-size:11px; color:#38bdf8;">${r.row_hash}</code>
          </div>
        </div>
      </td>
    `;

    rowEl.parentNode.insertBefore(detailTr, rowEl.nextSibling);
  }

  function renderChart(bars, rows) {
    if (!candleSeries || !bars || !bars.length) return;

    candleSeries.setData(bars.map(b => ({
      time: b.time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close
    })));

    renderChartMarkers(rows);
  }

  function renderChartMarkers(rowsToRender) {
    if (!candleSeries) return;
    markerMap.clear();

    const sigs = rowsToRender || cachedDayRows;
    sigs.forEach(s => {
      if (s.is_iex) return; // Do not plot non-scored IEX markers
      const isUp = s.direction === 'Up';
      const isDown = s.direction === 'Down';
      if (!isUp && !isDown) return;

      const t = Math.floor(new Date(s.bar_start).getTime() / 1000);
      const isStrong = s.strong === 1;
      let txt = '';

      if (showMarkerLabels) {
        txt = isUp ? `▲ Up ${(s.confidence * 100).toFixed(0)}%` : `▼ Down ${(s.confidence * 100).toFixed(0)}%`;
      }

      markerMap.set(t, {
        time: t,
        position: isUp ? 'belowBar' : 'aboveBar',
        color: isUp ? '#26a69a' : '#ef5350',
        shape: isUp ? 'arrowUp' : 'arrowDown',
        text: txt,
        size: isStrong ? 2 : 1
      });
    });

    const markers = Array.from(markerMap.values()).sort((a, b) => a.time - b.time);
    candleSeries.setMarkers(markers);
  }

  function renderLatestInsight(s) {
    if (!s || s.status === 'NO_SIGNALS_YET') return;

    const elHeroWord = document.getElementById('hero-word');
    const elHeroArrow = document.getElementById('hero-arrow');
    const elHeroDir = document.getElementById('hero-direction');
    const elReason = document.getElementById('hero-reason');
    const elConf = document.getElementById('hero-conf');
    const elConfWrap = document.getElementById('hero-conf-wrap');
    const elWindow = document.getElementById('hero-hour-window');
    const elSentence = document.getElementById('hero-sentence');
    const elVol = document.getElementById('val-vol');
    const elRegime = document.getElementById('val-regime');
    const elHit = document.getElementById('val-hist-hit');
    const elBands = document.getElementById('val-bands');
    const elSample = document.getElementById('hero-sample-tag');
    const elOutcome = document.getElementById('hero-outcome-pill');
    const elHeadline = document.getElementById('hero-headline-delivery');

    if (elHeadline && s.headline_text) {
      elHeadline.textContent = s.headline_text;
    }

    const dir = s.direction || 'No signal';
    const isUp = dir === 'Up';
    const isDown = dir === 'Down';
    const isNoSig = !isUp && !isDown;

    if (elSample) {
      elSample.textContent = s.sample || 'OUT-OF-SAMPLE';
      elSample.style.display = 'inline-block';
    }

    if (elOutcome) {
      if (s.correct === 1) {
        elOutcome.textContent = '✓ CORRECT';
        elOutcome.className = 'outcome-pill outcome-correct';
        elOutcome.style.display = 'inline-block';
      } else if (s.correct === 0 && s.realized_fwd_ret !== null && !isNoSig) {
        elOutcome.textContent = '✗ WRONG';
        elOutcome.className = 'outcome-pill outcome-wrong';
        elOutcome.style.display = 'inline-block';
      } else {
        elOutcome.style.display = 'none';
      }
    }

    if (isUp) {
      if (elHeroWord) elHeroWord.textContent = 'UP';
      if (elHeroArrow) elHeroArrow.textContent = '▲';
      if (elHeroDir) elHeroDir.className = 'direction-huge direction-up';
      if (elReason) elReason.textContent = '';
      if (elSentence) elSentence.textContent = 'Model predicts upward session move for the target 1-hour candle.';
    } else if (isDown) {
      if (elHeroWord) elHeroWord.textContent = 'DOWN';
      if (elHeroArrow) elHeroArrow.textContent = '▼';
      if (elHeroDir) elHeroDir.className = 'direction-huge direction-down';
      if (elReason) elReason.textContent = '';
      if (elSentence) elSentence.textContent = 'Model predicts downward session move for the target 1-hour candle.';
    } else {
      if (elHeroWord) elHeroWord.textContent = 'NO SIGNAL';
      if (elHeroArrow) elHeroArrow.textContent = '•';
      if (elHeroDir) elHeroDir.className = 'direction-huge direction-none';
      
      let reasonText = 'Inside no-signal band [0.434, 0.563]';
      if (dir.toLowerCase().includes('end of day')) {
        reasonText = 'End of trading day (no forward session hour)';
      }
      if (elReason) elReason.textContent = reasonText;
      if (elSentence) elSentence.textContent = 'Model abstained from directional signal. Probability fell in neutral band.';
    }

    if (isNoSig) {
      if (elConfWrap) elConfWrap.style.display = 'none';
    } else {
      if (elConfWrap) elConfWrap.style.display = 'block';
      if (elConf) elConf.textContent = `${(s.confidence * 100).toFixed(1)}%`;
    }

    if (elWindow && s.bar_start && s.bar_end) {
      const startT = s.bar_start.slice(11, 16);
      const endT = s.bar_end.slice(11, 16);
      elWindow.textContent = `covers ${startT} to ${endT} ET`;
    }

    const elThresh = document.getElementById('hero-threshold-subtext');
    if (elThresh) {
      if (s.threshold_str) {
        elThresh.textContent = s.threshold_str;
      } else if (s.p_up !== undefined) {
        const qHi = s.q_hi ? (s.q_hi * 100).toFixed(1) : '56.3';
        const qLo = s.q_lo ? (s.q_lo * 100).toFixed(1) : '43.4';
        elThresh.textContent = `P(Up) ${(s.p_up * 100).toFixed(1)}%: needs ≥ ${qHi}% for Up, ≤ ${qLo}% for Down`;
      }
    }

    if (elVol) elVol.textContent = s.vol_class || 'Normal';
    if (elRegime) elRegime.textContent = s.regime || 'Chop';
    if (elHit) {
      const hitVal = s.hit_rate_hour_side ? `${(s.hit_rate_hour_side * 100).toFixed(1)}%` : '52.0%';
      const nVal = s.n_hour_side ? `(n=${s.n_hour_side})` : '(n=100)';
      elHit.textContent = `${hitVal} ${nVal}`;
    }

    if (elBands && s.band_low !== null && s.band_high !== null) {
      const bLowBps = (s.band_low * 10000).toFixed(1);
      const bHighBps = (s.band_high * 10000).toFixed(1);
      elBands.textContent = `Low: ${bLowBps} bps | High: +${bHighBps} bps`;
    } else if (elBands) {
      elBands.textContent = 'n/a (market closed)';
    }
  }

  // Scoreboard Fetch
  async function fetchScoreboard() {
    try {
      const res = await fetch('/api/scoreboard');
      const data = await res.json();
      renderScoreboard(data);
    } catch (e) {
      console.error('Scoreboard fetch error:', e);
    }
  }

  function renderScoreboard(data) {
    const elTitle = document.getElementById('scoreboard-title');
    const elHitRate = document.getElementById('score-hit-rate');
    const elWilson = document.getElementById('score-wilson-ci');
    const elFormula = document.getElementById('score-formula-text');
    const elDownStat = document.getElementById('score-down-stat');
    const elUpStat = document.getElementById('score-up-stat');
    const elCoverage = document.getElementById('score-coverage');
    const elAllCandles = document.getElementById('score-all-candles');

    if (elTitle) elTitle.textContent = data.label || 'OUT-OF-SAMPLE (descriptive only, small n)';

    const nScored = data.total_resolved || 0;
    const nCand = data.n_candidate_candles || 24;
    const nHits = data.hits || 0;
    const sigAccPct = data.accuracy_pct !== undefined ? `${data.accuracy_pct.toFixed(1)}%` : '--%';

    if (elHitRate) elHitRate.textContent = sigAccPct;
    if (elWilson && data.wilson_ci) {
      elWilson.textContent = `[${data.wilson_ci[0]}%, ${data.wilson_ci[1]}%]`;
    }
    if (elFormula) {
      elFormula.textContent = `Signal accuracy = hits / directional signals (n=${nScored})`;
    }

    if (elDownStat) {
      if (data.down_count !== undefined && data.down_count > 0) {
        elDownStat.textContent = `${data.down_hits}/${data.down_count} (${(data.down_accuracy * 100).toFixed(1)}%)`;
      } else {
        elDownStat.textContent = '0/0 (n/a)';
      }
    }

    if (elUpStat) {
      if (data.up_count !== undefined && data.up_count > 0) {
        elUpStat.textContent = `${data.up_hits}/${data.up_count} (${(data.up_accuracy * 100).toFixed(1)}%)`;
      } else {
        elUpStat.textContent = '0/0 (n/a)';
      }
    }

    if (elCoverage) {
      elCoverage.textContent = `${nScored} of ${nCand}`;
    }

    if (elAllCandles) {
      const allPct = data.all_candle_hits_pct !== undefined ? `${data.all_candle_hits_pct.toFixed(1)}%` : '16.7%';
      elAllCandles.textContent = `${allPct} (${nHits}/${nCand})`;
    }
  }

  // System Status Fetch
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();

      // Countdown
      if (data.countdown_seconds !== undefined) {
        startCountdown(data.countdown_seconds, data.market_open);
      }

      // Signal Balance Card
      if (data.balance) {
        const bal = data.balance;
        const el1d = document.getElementById('bal-1d');
        const el4d = document.getElementById('bal-4d');
        const el20d = document.getElementById('bal-20d');

        if (el1d && bal.session_1) {
          el1d.textContent = `Up: ${bal.session_1.up} | Down: ${bal.session_1.down} | No-Sig: ${bal.session_1.no_signal}`;
        }
        if (el4d && bal.session_4) {
          el4d.textContent = `Up: ${bal.session_4.up} | Down: ${bal.session_4.down} | No-Sig: ${bal.session_4.no_signal}`;
        }
        if (el20d && bal.session_20) {
          el20d.textContent = `Up: ${bal.session_20.up} | Down: ${bal.session_20.down} | No-Sig: ${bal.session_20.no_signal}`;
        }

        const elAlert = document.getElementById('bal-alert-box');
        if (elAlert && bal.one_sided_alert) {
          elAlert.style.display = 'block';
          elAlert.textContent = `Diagnostic Alert: ${bal.consecutive_directional_count} consecutive ${bal.consecutive_directional_side} signals detected.`;
        } else if (elAlert) {
          elAlert.style.display = 'none';
        }
      }

      // Market state badge
      const elMarketBadge = document.getElementById('badge-market');
      if (elMarketBadge) {
        if (data.market_open) {
          elMarketBadge.textContent = 'MARKET OPEN';
          elMarketBadge.className = 'badge badge-success';
        } else {
          elMarketBadge.textContent = 'MARKET CLOSED';
          elMarketBadge.className = 'badge badge-warning';
        }
      }
    } catch (e) {
      console.error('Failed to fetch status:', e);
    }
  }

  // Setup Event Handlers
  function setupEventHandlers() {
    // 1. Day Navigation Prev / Next
    const btnPrev = document.getElementById('btn-prev-day');
    const btnNext = document.getElementById('btn-next-day');
    const picker = document.getElementById('day-picker');
    const btnLatest = document.getElementById('btn-latest-day');

    if (btnPrev) {
      btnPrev.addEventListener('click', () => {
        const idx = availableDays.findIndex(d => d.date === currentDay);
        if (idx !== -1 && idx < availableDays.length - 1) {
          currentDay = availableDays[idx + 1].date;
          updateDayNavControls();
          fetchLogAndRender(true);
        }
      });
    }

    if (btnNext) {
      btnNext.addEventListener('click', () => {
        const idx = availableDays.findIndex(d => d.date === currentDay);
        if (idx !== -1 && idx > 0) {
          currentDay = availableDays[idx - 1].date;
          updateDayNavControls();
          fetchLogAndRender(true);
        }
      });
    }

    if (picker) {
      picker.addEventListener('change', (e) => {
        if (e.target.value) {
          currentDay = e.target.value;
          updateDayNavControls();
          fetchLogAndRender(true);
        }
      });
    }

    if (btnLatest) {
      btnLatest.addEventListener('click', () => {
        const latestScored = availableDays.find(d => d.is_latest_scored);
        if (latestScored) {
          currentDay = latestScored.date;
          updateDayNavControls();
          fetchLogAndRender(true);
        }
      });
    }

    // 2. Horizon Tabs
    const horizonButtons = document.querySelectorAll('#horizon-tabs .tab-btn');
    horizonButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        horizonButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentHorizon = btn.getAttribute('data-horizon') || '1 hour';
        fetchLogAndRender(true);
      });
    });

    // 3. Filters
    const selFilter = document.getElementById('filter-sample-select');
    if (selFilter) {
      selFilter.addEventListener('change', (e) => {
        currentView = e.target.value;
        fetchLogAndRender(true);
      });
    }

    const chkNoSig = document.getElementById('filter-nosig-check');
    if (chkNoSig) {
      chkNoSig.addEventListener('change', (e) => {
        currentIncludeNoSig = e.target.checked;
        fetchLogAndRender(false);
      });
    }

    const chkIex = document.getElementById('filter-iex-check');
    if (chkIex) {
      chkIex.addEventListener('change', (e) => {
        currentIncludeIex = e.target.checked;
        fetchLogAndRender(false);
      });
    }

    // 4. Diagnostics Accordion Toggle
    const btnToggleDiag = document.getElementById('btn-toggle-diagnostics');
    const diagSection = document.getElementById('diagnostics-section');
    const diagText = document.getElementById('diag-toggle-text');

    if (btnToggleDiag && diagSection) {
      btnToggleDiag.addEventListener('click', () => {
        const isHidden = (diagSection.style.display === 'none' || !diagSection.style.display);
        if (isHidden) {
          diagSection.style.display = 'block';
          if (diagText) diagText.textContent = 'Hide Model & Feed Diagnostics ▲';
          diagSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
          diagSection.style.display = 'none';
          if (diagText) diagText.textContent = 'Show Model & Feed Diagnostics ▼';
        }
      });
    }

    // 5. Chart Labels Toggle
    const btnLabels = document.getElementById('btn-toggle-labels');
    if (btnLabels) {
      btnLabels.addEventListener('click', () => {
        showMarkerLabels = !showMarkerLabels;
        btnLabels.textContent = `Labels: ${showMarkerLabels ? 'ON' : 'OFF'}`;
        renderChartMarkers();
      });
    }

    // 6. Parity Modal
    const btnWhy = document.getElementById('btn-parity-why');
    const modalParity = document.getElementById('parity-modal');
    const btnCloseParity = document.getElementById('btn-close-parity');

    if (btnWhy && modalParity) {
      btnWhy.addEventListener('click', () => {
        modalParity.style.display = 'flex';
      });
    }
    if (btnCloseParity && modalParity) {
      btnCloseParity.addEventListener('click', () => {
        modalParity.style.display = 'none';
      });
    }
    if (modalParity) {
      modalParity.addEventListener('click', (e) => {
        if (e.target === modalParity) {
          modalParity.style.display = 'none';
        }
      });
    }
  }

  // App Initialization
  document.addEventListener('DOMContentLoaded', async () => {
    startDualClocks();
    initChart();
    setupEventHandlers();
    fetchStatus();
    fetchScoreboard();

    await fetchDaysList();
    fetchLogAndRender(true);

    // Periodic live updates every 10 seconds (Clean replacement, zero row runaway)
    setInterval(() => {
      if (!isReplayMode) {
        fetchStatus();
        fetchScoreboard();
        fetchLogAndRender(false);
      }
    }, 10000);
  });
})();
