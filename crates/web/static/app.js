/* App JS — shared helpers.
 *
 * `ParetoChart` builds a Pareto visualisation with one-click
 * switching between a 3D view (all three NSGA-III objectives at
 * once) and three 2D pairwise projections. Used by hpo_detail.html
 * (live and completed) and hpo_progress.html.
 *
 * Axis convention (NSGA-III directions from the workflow):
 *   OK@1%       — maximise (higher is better)
 *   param_count — minimise (lower is better)
 *   max_error   — minimise (lower is better)
 *
 * Trial shape (all fields optional except ok / param_count / max_error):
 *   { ok, param_count, max_error,
 *     trial_number, model_type,
 *     is_pareto, is_feasible,
 *     label, hover }
 */
(function(global) {
  if (typeof Plotly === 'undefined') return;

  // ── Trace configuration ─────────────────────────────────────────
  // Six buckets: {real|complex} × {infeasible|feasible|pareto}. A
  // study running only one model type keeps half of them empty,
  // which is fine — empty traces are filtered before plotting.
  var TRACE_CONFIGS = [
    { key: 'real_infeasible',    name: 'Real infeasible',    color: 'rgba(148,163,184,0.30)', size: 4,  symbol: 'x',       d3Symbol: 'x',            line: null },
    { key: 'complex_infeasible', name: 'Complex infeasible', color: 'rgba(192,132,252,0.30)', size: 4,  symbol: 'x',       d3Symbol: 'x',            line: null },
    { key: 'real_feasible',      name: 'Real feasible',      color: 'rgba(100,116,139,0.55)', size: 6,  symbol: 'circle',  d3Symbol: 'circle',       line: null },
    { key: 'complex_feasible',   name: 'Complex feasible',   color: 'rgba(147,51,234,0.45)',  size: 6,  symbol: 'circle',  d3Symbol: 'circle',       line: null },
    { key: 'real_pareto',        name: 'Real Pareto',        color: '#16a34a',                size: 9,  symbol: 'diamond', d3Symbol: 'diamond',      line: { color: '#15803d', width: 1 } },
    { key: 'complex_pareto',     name: 'Complex Pareto',     color: '#7c3aed',                size: 9,  symbol: 'diamond', d3Symbol: 'diamond',      line: { color: '#6d28d9', width: 1 } },
  ];

  // ── View catalogue ──────────────────────────────────────────────
  // Each view picks which objective fields map to which axis; `3d`
  // uses all three, the three 2D views drop one axis. The `better`
  // arrows in the axis labels help readers recall which direction
  // is the desirable one for each objective.
  var VIEWS = {
    '3d': {
      label: '3D · all objectives',
      dims: 3,
      x: { field: 'ok',          title: 'OK@1% (%) ▲ better' },
      y: { field: 'param_count', title: 'Param count ▼ better' },
      z: { field: 'max_error',   title: 'Max error (%) ▼ better' },
    },
    'ok-params': {
      label: 'OK@1% × Params',
      dims: 2,
      x: { field: 'ok',          title: 'OK@1% (%) ▲ better' },
      y: { field: 'param_count', title: 'Param count ▼ better' },
    },
    'ok-maxerr': {
      label: 'OK@1% × Max err',
      dims: 2,
      x: { field: 'ok',         title: 'OK@1% (%) ▲ better' },
      y: { field: 'max_error',  title: 'Max error (%) ▼ better' },
    },
    'params-maxerr': {
      label: 'Params × Max err',
      dims: 2,
      x: { field: 'param_count', title: 'Param count ▼ better' },
      y: { field: 'max_error',   title: 'Max error (%) ▼ better' },
    },
  };
  var VIEW_ORDER = ['3d', 'ok-params', 'ok-maxerr', 'params-maxerr'];

  // ── Helpers ─────────────────────────────────────────────────────
  function bucketKey(trial) {
    var mtype = (trial.model_type === 'complex') ? 'complex' : 'real';
    var status = trial.is_pareto ? 'pareto'
               : (trial.is_feasible ? 'feasible' : 'infeasible');
    return mtype + '_' + status;
  }

  function defaultLabel(t) {
    var mtype = t.model_type || 'real';
    var parts = ['#' + (t.trial_number != null ? t.trial_number : '?'),
                 '[' + mtype + ']'];
    if (t.param_count != null) parts.push('params=' + t.param_count);
    if (t.max_error != null && isFinite(t.max_error)) {
      parts.push('max_err=' + Number(t.max_error).toFixed(2) + '%');
    }
    return parts.join(' ');
  }

  function groupTrials(trials) {
    // Group into bucket_key → arrays-of-axis-values. Each trial
    // contributes its numeric coordinates + a text label + the
    // original trial_number for click-through. Missing / non-finite
    // objectives are skipped at render time (per-view).
    var groups = {};
    TRACE_CONFIGS.forEach(function(tc) {
      groups[tc.key] = { ok: [], param_count: [], max_error: [],
                         text: [], trials: [] };
    });
    trials.forEach(function(t) {
      var key = bucketKey(t);
      var g = groups[key];
      if (!g) return;
      g.ok.push(t.ok != null ? t.ok : null);
      g.param_count.push(t.param_count != null ? t.param_count : null);
      g.max_error.push(t.max_error != null ? t.max_error : null);
      g.text.push(t.label || defaultLabel(t));
      g.trials.push(t.trial_number != null ? t.trial_number : null);
    });
    return groups;
  }

  function buildTraces(groups, view) {
    // Build one Plotly trace per non-empty bucket in the current
    // view. 3D traces use `scatter3d` + x/y/z; 2D traces use
    // `scatter` + x/y. Markers re-use TRACE_CONFIGS so the
    // pareto/feasible/infeasible legend stays consistent across
    // views.
    var traces = [];
    var hoverTemplate = view.dims === 3
      ? '%{text}<br>%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<br>%{zaxis.title.text}: %{z}<extra>%{data.name}</extra>'
      : '%{text}<br>%{xaxis.title.text}: %{x}<br>%{yaxis.title.text}: %{y}<extra>%{data.name}</extra>';

    TRACE_CONFIGS.forEach(function(tc) {
      var g = groups[tc.key];
      if (!g || g.ok.length === 0) return;

      // Materialise numeric axes from the view spec; keep original
      // indexing so hover text / customdata line up row-for-row.
      var xs = g[view.x.field];
      var ys = g[view.y.field];
      var zs = view.dims === 3 ? g[view.z.field] : null;

      var trace;
      if (view.dims === 3) {
        trace = {
          x: xs, y: ys, z: zs,
          text: g.text,
          customdata: g.trials,
          name: tc.name,
          type: 'scatter3d',
          mode: 'markers',
          marker: {
            color: tc.color,
            size: Math.max(3, tc.size - 1),  // 3D points read larger
            symbol: tc.d3Symbol,
            line: tc.line || undefined,
            opacity: 0.95,
          },
          hovertemplate: hoverTemplate,
        };
      } else {
        trace = {
          x: xs, y: ys,
          text: g.text,
          customdata: g.trials,
          name: tc.name,
          type: 'scatter',
          mode: 'markers',
          marker: {
            color: tc.color,
            size: tc.size,
            symbol: tc.symbol,
            line: tc.line || undefined,
          },
          hovertemplate: hoverTemplate,
        };
      }
      traces.push(trace);
    });
    return traces;
  }

  function buildLayout(view, height) {
    var common = {
      margin: view.dims === 3
        ? { t: 6, r: 8, b: 6, l: 8 }
        : { t: 10, r: 24, b: 55, l: 60 },
      font: { family: "'JetBrains Mono', monospace", size: 11 },
      showlegend: true,
      legend: { orientation: 'h', y: -0.10, x: 0.5, xanchor: 'center',
                font: { size: 10 } },
    };
    if (view.dims === 3) {
      // Camera chosen so "best corner" (high OK, low params, low
      // max_error) sits towards the viewer. Users can still rotate;
      // this is just the landing frame.
      common.scene = {
        xaxis: { title: { text: view.x.title } },
        yaxis: { title: { text: view.y.title } },
        zaxis: { title: { text: view.z.title } },
        camera: { eye: { x: 1.8, y: -1.6, z: 1.1 } },
        aspectmode: 'cube',
      };
    } else {
      common.xaxis = { title: { text: view.x.title, standoff: 8 } };
      common.yaxis = { title: { text: view.y.title } };
    }
    return common;
  }

  function renderButtons(container, onSelect, current) {
    // Minimal button group above the chart. Re-rendered each time
    // the active view changes so the pressed state stays in sync.
    container.innerHTML = '';
    container.style.display = 'flex';
    container.style.flexWrap = 'wrap';
    container.style.gap = '4px';
    container.style.marginBottom = '6px';
    VIEW_ORDER.forEach(function(key) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'btn btn-sm' + (key === current ? ' btn-primary' : ' btn-secondary');
      btn.textContent = VIEWS[key].label;
      btn.style.fontSize = '11px';
      btn.style.padding = '2px 8px';
      btn.addEventListener('click', function() { onSelect(key); });
      container.appendChild(btn);
    });
  }

  // ── Public API ──────────────────────────────────────────────────
  global.ParetoChart = {
    /**
     * Create a Pareto chart with 3D + three 2D views.
     *
     * `node` — DOM element that will host the chart + button bar.
     * `trials` — initial array of trial objects (may be empty).
     * `options`:
     *   height            (number, default 420)  pixel height
     *   initialView       (string, default '3d') one of VIEW_ORDER
     *   filenamePrefix    (string, default 'pareto')
     *   onTrialClick      (fn(trialNumber))      invoked when a point is clicked
     *   showViewButtons   (bool, default true)   hide for compact live views
     *
     * Returns a controller with:
     *   setView(view)     switch to a view by key
     *   replace(trials)   replace full dataset
     *   addTrial(trial)   append one trial (live SSE path)
     *   destroy()         tear down Plotly + buttons
     */
    create: function(node, trials, options) {
      options = options || {};
      var height = options.height || 420;
      var filenamePrefix = options.filenamePrefix || 'pareto';

      // Shell: [button bar] + [plot div]
      node.innerHTML = '';
      var btnBar = document.createElement('div');
      var plot = document.createElement('div');
      plot.style.width = '100%';
      plot.style.height = height + 'px';
      if (options.showViewButtons !== false) node.appendChild(btnBar);
      node.appendChild(plot);

      var state = {
        view: options.initialView && VIEWS[options.initialView] ? options.initialView : '3d',
        trials: trials ? trials.slice() : [],
      };

      function render() {
        var view = VIEWS[state.view];
        var groups = groupTrials(state.trials);
        var traces = buildTraces(groups, view);
        var layout = buildLayout(view, height);
        Plotly.react(plot, traces, layout, {
          responsive: true,
          displaylogo: false,
          toImageButtonOptions: {
            format: 'png',
            filename: filenamePrefix + '_' + state.view,
            scale: 2,
          },
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        });
        if (options.showViewButtons !== false) {
          renderButtons(btnBar, function(k) {
            state.view = k;
            render();
          }, state.view);
        }
      }

      render();

      if (typeof options.onTrialClick === 'function') {
        // `plotly_click` fires once the plot is initialised; `on`
        // only exists after `Plotly.react` resolves, which it does
        // synchronously on CPU.
        plot.on('plotly_click', function(data) {
          if (!data || !data.points || data.points.length === 0) return;
          var trialNum = data.points[0].customdata;
          if (trialNum != null) options.onTrialClick(trialNum);
        });
      }

      return {
        setView: function(key) {
          if (!VIEWS[key]) return;
          state.view = key;
          render();
        },
        replace: function(newTrials) {
          state.trials = newTrials ? newTrials.slice() : [];
          render();
        },
        addTrial: function(trial) {
          state.trials.push(trial);
          render();
        },
        destroy: function() {
          Plotly.purge(plot);
          node.innerHTML = '';
        },
      };
    },
  };
})(window);
