// Lightweight Charts & Real-time Live Trading Indicator Dashboard
(function () {
  'use strict';

  let chart = null;
  let candleSeries = null;
  let bandHighLine = null;
  let bandLowLine = null;
  let countdownSeconds = 0;
  let countdownTimer = null;

  // Track plotted markers by bar timestamp so past markers never change between refreshes
  const markerMap = new Map();

  function initChart() {
    const container = document.getElementById('tv-chart');
    if (!container) return;

    chart = LightweightCharts.createChart(container, {
      layout: {
        background: { color: '#12151c' },
        textColor: '#8b95a8',
        fontSize: 12
      },
      grid: {
        vertLines: { color: 'rgba(46, 53, 71, 0.4)' },
        horzLines: { color: 'rgba(46, 53, 71, 0.4)' }
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: '#2e3547'
      },
      rightPriceScale: {
        borderColor: '#2e3547',
        scaleMargins: { top: 0.1, bottom: 0.1 }
      }
    });

    candleSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: true,
      borderUpColor: '#26a69a',
      borderDownColor: '#ef5350',
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350'
    });

    // Quantile Bands as reference lines
    bandHighLine = chart.addLineSeries({
      color: 'rgba(38, 166, 154, 0.6)',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      title: 'Band High'
    });

    bandLowLine = chart.addLineSeries({
      color: 'rgba(239, 83, 80, 0.6)',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      title: 'Band Low'
    });

    window.addEventListener('resize', () => {
      chart.resize(container.clientWidth, container.clientHeight);
    });
  }

  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      updateHeaderBadges(data);
      updateBanners(data.banners || []);
      if (data.server_time_et) {
        document.getElementById('footer-time').textContent = `Server: ${data.server_time_et.slice(11, 19)} ET`;
      }
    } catch (e) {
      console.error('Status fetch error:', e);
    }
  }

  function updateHeaderBadges(status) {
    const bMarket = document.getElementById('badge-market');
    if (status.market_open) {
      bMarket.textContent = 'MARKET OPEN';
      bMarket.className = 'badge badge-live';
    } else {
      bMarket.textContent = 'MARKET CLOSED';
      bMarket.className = 'badge badge-closed';
    }

    const bSignals = document.getElementById('badge-signals');
    if (status.live_signals_enabled) {
      bSignals.textContent = 'SIGNALS ACTIVE';
      bSignals.className = 'badge badge-live';
    } else {
      bSignals.textContent = 'SIGNALS DISABLED';
      bSignals.className = 'badge badge-warning';
    }

    const bModel = document.getElementById('badge-model');
    bModel.textContent = `MODEL: ${status.model_hash}`;
  }

  function updateBanners(banners) {
    const container = document.getElementById('banner-container');
    container.innerHTML = '';
    banners.forEach((b) => {
      const div = document.createElement('div');
      div.className = 'banner-alert';
      div.textContent = b;
      container.appendChild(div);
    });
  }

  async function fetchBarsAndSignals() {
    try {
      const [barsRes, signalsRes, latestRes, scoreRes] = await Promise.all([
        fetch('/api/bars?limit=120'),
        fetch('/api/signals?limit=40'),
        fetch('/api/latest'),
        fetch('/api/scoreboard')
      ]);

      const bars = await barsRes.json();
      const signals = await signalsRes.json();
      const latest = await latestRes.json();
      const score = await scoreRes.json();

      renderChart(bars, signals);
      renderLatestInsight(latest);
      renderScoreboard(score);
      renderSignalsTable(signals);
    } catch (e) {
      console.error('Data fetch error:', e);
    }
  }

  function renderChart(bars, signals) {
    if (!candleSeries || !bars.length) return;

    // Filter forming candle styling
    candleSeries.setData(bars.map((b) => ({
      time: b.time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close
    })));

    // Associate signals to bars
    signals.forEach((s) => {
      const t = Math.floor(new Date(s.bar_start).getTime() / 1000);
      if (!markerMap.has(t)) {
        let shape = 'circle';
        let pos = 'inBar';
        let col = '#8b95a8';
        let txt = '';
        const isStrong = s.strong === 1;

        if (s.direction === 'Up') {
          shape = 'arrowUp';
          pos = 'belowBar';
          col = '#26a69a';
          txt = `▲ Up ${(s.confidence * 100).toFixed(0)}%${isStrong ? ' [STRONG]' : ''}`;
        } else if (s.direction === 'Down') {
          shape = 'arrowDown';
          pos = 'aboveBar';
          col = '#ef5350';
          txt = `▼ Down ${(s.confidence * 100).toFixed(0)}%${isStrong ? ' [STRONG]' : ''}`;
        } else {
          shape = 'circle';
          pos = 'aboveBar';
          col = '#8b95a8';
          txt = '•';
        }

        markerMap.set(t, {
          time: t,
          position: pos,
          color: col,
          shape: shape,
          text: txt,
          size: isStrong ? 2 : 1
        });
      }
    });

    const markers = Array.from(markerMap.values()).sort((a, b) => a.time - b.time);
    candleSeries.setMarkers(markers);
  }

  function renderLatestInsight(s) {
    if (!s || s.status === 'NO_SIGNALS_YET') return;

    const dirEl = document.getElementById('hero-direction');
    const confEl = document.getElementById('hero-conf');
    const strongPill = document.getElementById('strong-pill');

    dirEl.textContent = s.direction ? s.direction.toUpperCase() : 'NO SIGNAL';
    dirEl.className = 'direction-title';
    if (s.direction === 'Up') dirEl.classList.add('direction-up');
    else if (s.direction === 'Down') dirEl.classList.add('direction-down');
    else dirEl.classList.add('direction-none');

    confEl.textContent = s.confidence ? `${(s.confidence * 100).toFixed(1)}%` : '--%';
    strongPill.style.display = s.strong ? 'inline-block' : 'none';

    document.getElementById('val-vol').textContent = s.vol_class || '--';
    document.getElementById('val-regime').textContent = s.regime || '--';
    document.getElementById('val-hist-hit').textContent = s.hit_rate_hour_side
      ? `${(s.hit_rate_hour_side * 100).toFixed(1)}% (n=${s.n_hour_side || 100})`
      : '--%';
  }

  function renderScoreboard(score) {
    if (!score) return;
    document.getElementById('score-resolved').textContent = score.total_resolved || 0;
    document.getElementById('score-acc').textContent = score.accuracy !== null
      ? `${(score.accuracy * 100).toFixed(1)}%`
      : '--%';
  }

  function renderSignalsTable(signals) {
    const tbody = document.getElementById('signals-tbody');
    if (!tbody) return;

    if (!signals.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:#8b95a8;">No signals recorded yet.</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    signals.slice(0, 15).forEach((s) => {
      const tr = document.createElement('tr');
      const timeStr = s.bar_start.slice(11, 16);
      const confStr = `${(s.confidence * 100).toFixed(0)}%`;

      let outcomeHtml = '<span class="res-pending">Pending</span>';
      if (s.correct === 1) outcomeHtml = '<span class="res-correct">Correct ✓</span>';
      else if (s.correct === 0) outcomeHtml = '<span class="res-wrong">Wrong ✗</span>';

      const dirClass = s.direction === 'Up' ? 'res-correct' : (s.direction === 'Down' ? 'res-wrong' : '');
      const strongTag = s.strong ? ' <small style="color:#f59e0b;">[S]</small>' : '';

      tr.innerHTML = `
        <td>${timeStr}</td>
        <td class="${dirClass}">${s.direction}${strongTag}</td>
        <td>${confStr}</td>
        <td>${outcomeHtml}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function startCountdown(seconds) {
    countdownSeconds = seconds;
    if (countdownTimer) clearInterval(countdownTimer);

    function tick() {
      if (countdownSeconds > 0) {
        countdownSeconds--;
        const mins = Math.floor(countdownSeconds / 60);
        const secs = countdownSeconds % 60;
        document.getElementById('val-countdown').textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
      } else {
        document.getElementById('val-countdown').textContent = '00:00';
      }
    }

    tick();
    countdownTimer = setInterval(tick, 1000);
  }

  function setupSSE() {
    if (window.EventSource) {
      const sse = new EventSource('/api/stream');
      sse.addEventListener('tick', (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.countdown_seconds !== undefined) {
            startCountdown(data.countdown_seconds);
          }
          if (data.latest_signal) {
            renderLatestInsight(data.latest_signal);
          }
        } catch (err) {
          console.error('SSE parse error:', err);
        }
      });
      sse.onerror = () => {
        console.warn('SSE disconnected. Polling fallback active.');
      };
    }
  }

  // Lifecycle boot
  document.addEventListener('DOMContentLoaded', () => {
    initChart();
    fetchStatus();
    fetchBarsAndSignals();
    setupSSE();

    // 5-second polling fallback
    setInterval(() => {
      fetchStatus();
      fetchBarsAndSignals();
    }, 5000);
  });
})();
