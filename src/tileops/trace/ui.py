"""Self-contained Plotly HTML timeline viewer for decoded trace events.

Turns the flat ``Slice`` / ``Instant`` list from
``tileops.trace.decode.decode`` into a single, self-contained ``.html`` file
that renders one horizontal-bar Gantt timeline per CTA. The viewer matches the
look of the reference GPU-pipeline timeline in ``docs/plans/reference_fa3_pipeline.html``:
warm ``#f3ede1`` paper, monospace fonts, overlaid bars with light separator lines,
a categorical lane axis, an x axis labelled in SM cycles, and a horizontal legend
pinned at the bottom.

This module is pure Python — it emits the Plotly figure dicts as embedded JSON
plus a small JS shim and loads Plotly itself from a CDN. It imports neither torch
nor plotly, so it stays importable in any environment.

Layout per CTA (``_figure_for_cta``):
    * **Lanes** are auto-derived from the data: the distinct ``(gid, lane)`` pairs
      that carry events in the CTA, sorted by ``(gid, lane)``. Each lane's label is
      ``"<group> / <lane>"``. Plotly draws a categorical ``categoryarray``
      bottom-up, so the array is the lane labels in **reverse** sorted order — the
      first ``(gid, lane)`` (producer / group 0, ``main`` lane) lands last in the
      array and therefore renders at the top, matching the reference (its
      ``categoryarray`` likewise lists the visually-bottom lane first).
    * **Traces** are grouped by event ``name`` across all lanes — one bar trace per
      name, each with a stable color picked deterministically from a warm palette by
      sorted name, so a name keeps its color across CTAs. ``Instant`` markers are
      grouped likewise into thin (1% of span) bar traces.
    * **Flows** (declared ``(src_name, dst_name)`` pairs resolved per CTA by
      ``tileops.trace.decode.compute_flows``) become one ``annotations`` arrow
      per ``FlowEdge``, from the source slice's end lane
      to the destination slice's start lane.

Interaction is horizontal-only zoom + pan: ``dragmode:"pan"`` with
``xaxis.fixedrange:false`` and ``yaxis.fixedrange:true``, plus a Plotly config that
keeps ``scrollZoom`` on and strips the vertical/box zoom buttons. With the y axis
fixed, scroll-zoom only stretches x and pan only moves x.
"""

import json

from .decode import Instant, Slice, compute_flows

__all__ = ["export_timeline_html"]

# Warm palette seeded from the reference timeline, then extended with further warm
# / muted tones. Colors are assigned to event names deterministically by sorted
# order so a name keeps its color across CTAs.
_PALETTE = [
    "#c43c19",  # red-orange (reference: softmax)
    "#2f5d8c",  # steel blue (reference: QK wgmma)
    "#1a6e63",  # teal (reference: PV wgmma)
    "#d8a13a",  # amber (reference: rescale)
    "#7d5ba6",  # violet (reference: wait peer / barrier)
    "#7f97a6",  # slate (reference: TMA load)
    "#9a8f7a",  # taupe (reference: wgmma drain)
    "#5f6d76",  # gunmetal (reference: store)
    "#b5651d",  # ochre
    "#4a7c59",  # moss
    "#8c4a6b",  # plum
    "#3d6b8c",  # denim
    "#a6803a",  # bronze
    "#6b6f5f",  # olive grey
    "#9c5d3a",  # terracotta
    "#5a6b8c",  # dusty blue
]

# Annotation arrow color for dag / barrier dependency edges (reference violet).
_ARROW_COLOR = "#7d5ba6"

# Shared aesthetic constants (match the reference).
_PAPER_BG = "#f3ede1"
_GRID_COLOR = "#d8cfbb"
_BAR_LINE_COLOR = "#f3ede1"
_FONT_FAMILY = "ui-monospace, SFMono-Regular, monospace"
_INK = "#191a16"


def _lane_label(gid: int, lane: int, group_id_to_name: dict, lane_id_to_name: dict) -> str:
    """Build a lane's y-axis label ``"<group> / <lane>"``.

    Args:
        gid: Logical work-group id of the lane.
        lane: Interned render sub-lane id.
        group_id_to_name: ``gid -> name`` map; falls back to ``g<gid>``.
        lane_id_to_name: ``lane_id -> name`` map; falls back to the id.

    Returns:
        The lane label string.
    """
    gname = group_id_to_name.get(gid, f"g{gid}")
    lname = lane_id_to_name.get(lane, str(lane))
    return f"{gname} / {lname}"


def _assign_colors(names: list) -> dict:
    """Map each event name to a stable palette color by sorted order.

    Args:
        names: All distinct event names appearing across every CTA.

    Returns:
        ``name -> hex color`` dict; the same name always gets the same color.
    """
    return {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(sorted(names))}


def _text_color(hex_color: str) -> str:
    """Pick black or white inside-bar text for legible contrast on ``hex_color``.

    Args:
        hex_color: A ``#rrggbb`` bar fill color.

    Returns:
        ``"#191a16"`` on light fills, ``"#ffffff"`` on dark fills.
    """
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return _INK if luminance > 0.6 else "#ffffff"


def _ns(cy: float, sm_clock_ghz: float) -> float:
    """Convert a cycle count to nanoseconds at the locked SM clock.

    ``cycles / GHz == ns`` (a GHz clock advances one cycle per ns).

    Args:
        cy: Cycle count.
        sm_clock_ghz: Locked SM clock in GHz.

    Returns:
        The duration in nanoseconds, rounded to one decimal.
    """
    return round(cy / sm_clock_ghz, 1)


def _bar_trace(name, color, lane_labels, bases, durs, texts, payloads, sm_clock_ghz,
               *, width, textangle):
    """Build one Plotly horizontal-bar trace for an event name.

    Args:
        name: Event name (legend entry).
        color: Bar fill color.
        lane_labels: y category per bar.
        bases: ``base`` (start cycle) per bar.
        durs: ``x`` (duration cycle) per bar.
        texts: Inside-bar text per bar.
        payloads: Raw payload per bar (for the hover ``customdata``).
        sm_clock_ghz: Locked SM clock in GHz (for the ~ns hover column).
        width: Bar thickness (Plotly y-units).
        textangle: Inside-text rotation in degrees.

    Returns:
        A Plotly trace dict.
    """
    customdata = [[name, "" if p is None else p, d, _ns(d, sm_clock_ghz)]
                  for d, p in zip(durs, payloads, strict=True)]
    return {
        "type": "bar",
        "orientation": "h",
        "name": name,
        "base": bases,
        "x": durs,
        "y": lane_labels,
        "text": texts,
        "textposition": "inside",
        "insidetextanchor": "middle",
        "constraintext": "inside",
        "cliponaxis": False,
        "textangle": textangle,
        "textfont": {"color": _text_color(color), "size": 11, "family": _FONT_FAMILY},
        "marker": {"color": color, "line": {"color": _BAR_LINE_COLOR, "width": 1.2}},
        "width": width,
        "showlegend": True,
        "customdata": customdata,
        "hovertemplate": ("<b>%{customdata[0]}</b>  payload %{customdata[1]}<br>"
                          "start %{base} cyc   dur %{x} cyc "
                          "(~%{customdata[3]} ns)<extra></extra>"),
    }


def _figure_for_cta(cta, events, group_id_to_name, lane_id_to_name, flows, color_map,
                    title, sm_clock_ghz):
    """Build the Plotly ``{data, layout}`` figure dict for one CTA.

    Args:
        cta: The flat CTA index.
        events: This CTA's ``Slice`` / ``Instant`` objects.
        group_id_to_name: ``gid -> name`` map for lane labels.
        lane_id_to_name: ``lane_id -> name`` map for lane labels.
        flows: Declared ``(src_name, dst_name)`` pairs, resolved to per-edge arrows
            for this CTA via ``tileops.trace.decode.compute_flows``.
        color_map: Global ``name -> color`` assignment (stable across CTAs).
        title: Base figure title; the CTA index is appended.
        sm_clock_ghz: Locked SM clock in GHz.

    Returns:
        A ``{"data": [...], "layout": {...}}`` dict ready for ``Plotly.react``.
    """
    slices = [e for e in events if isinstance(e, Slice)]
    instants = [e for e in events if isinstance(e, Instant)]
    # compute_flows pairs same-named slices in timestamp order; events are already
    # restricted to this CTA, so each declared flow yields its per-occurrence edges.
    edges = compute_flows(events, flows or [])

    # Lanes auto-derived: distinct (gid, lane) with events, sorted by (gid, lane).
    lane_pairs = sorted({(e.track[1], e.track[2]) for e in events})
    lane_labels = {(g, l): _lane_label(g, l, group_id_to_name, lane_id_to_name)
                   for (g, l) in lane_pairs}

    # x span for instant-bar width and the initial range.
    span = 0
    for s in slices:
        span = max(span, s.ts_cy + s.dur_cy)
    for e in instants:
        span = max(span, e.ts_cy)
    for edge in edges:
        span = max(span, edge.src_ts_cy, edge.dst_ts_cy)
    span = span or 1
    instant_w = max(1, span * 0.01)

    data = []

    # Slice traces, grouped by name.
    by_name: dict = {}
    for s in slices:
        by_name.setdefault(s.name, []).append(s)
    for name in sorted(by_name):
        items = by_name[name]
        text = [(f"{s.name}<br>{s.payload}" if s.payload is not None else s.name)
                for s in items]
        data.append(_bar_trace(
            name, color_map[name],
            [lane_labels[(s.track[1], s.track[2])] for s in items],
            [s.ts_cy for s in items],
            [s.dur_cy for s in items],
            text,
            [s.payload for s in items],
            sm_clock_ghz, width=0.62, textangle=0))

    # Instant traces (thin bars), grouped by name.
    inst_by_name: dict = {}
    for inst in instants:
        inst_by_name.setdefault(inst.name, []).append(inst)
    for name in sorted(inst_by_name):
        items = inst_by_name[name]
        data.append(_bar_trace(
            name, color_map[name],
            [lane_labels[(i.track[1], i.track[2])] for i in items],
            [i.ts_cy for i in items],
            [instant_w] * len(items),
            [name] * len(items),
            [i.payload for i in items],
            sm_clock_ghz, width=0.34, textangle=0))

    # Plotly draws categoryarray bottom-up; reverse the sorted lane order so the
    # first (gid, lane) — producer / group0 main — renders at the TOP.
    category_array = [lane_labels[p] for p in reversed(lane_pairs)]

    # Flow edges -> one annotation arrow each, from the source slice's end lane to
    # the destination slice's start lane. Each FlowEdge is a distinct occurrence,
    # so the arrows fan out instead of collapsing onto a shared endpoint.
    annotations = []
    for edge in edges:
        src_label = _lane_label(edge.src_track[1], edge.src_track[2], group_id_to_name,
                                lane_id_to_name)
        dst_label = _lane_label(edge.dst_track[1], edge.dst_track[2], group_id_to_name,
                                lane_id_to_name)
        src_ts = edge.src_ts_cy
        dst_ts = edge.dst_ts_cy
        # Arrow only, no text label: the line from src to dst already says it all.
        annotations.append({
            "x": dst_ts, "y": dst_label,
            "ax": src_ts, "ay": src_label,
            "xref": "x", "yref": "y", "axref": "x", "ayref": "y",
            "showarrow": True, "arrowhead": 2, "arrowwidth": 1.5, "arrowsize": 1.1,
            "arrowcolor": _ARROW_COLOR, "opacity": 0.85, "text": "",
        })

    layout = {
        "title": {
            "text": f"<b>{title} · CTA {cta}</b>",
            "font": {"size": 20, "family": _FONT_FAMILY},
            "x": 0.02, "xanchor": "left",
        },
        "barmode": "overlay",
        "bargap": 0.35,
        "paper_bgcolor": _PAPER_BG,
        "plot_bgcolor": _PAPER_BG,
        "font": {"family": _FONT_FAMILY, "color": _INK, "size": 13},
        "dragmode": "pan",
        "xaxis": {
            "title": {"text": "SM cycles (clock64)", "standoff": 10},
            "gridcolor": _GRID_COLOR,
            "zeroline": False,
            "range": [0, span * 1.02],
            "fixedrange": False,
            "showspikes": True,
            "spikemode": "across",
            "spikecolor": "#7c7565",
            "spikethickness": 1,
        },
        "yaxis": {
            "categoryorder": "array",
            "categoryarray": category_array,
            "fixedrange": True,
            "tickfont": {"size": 13, "family": _FONT_FAMILY},
            "automargin": True,
        },
        "legend": {"orientation": "h", "y": -0.30, "yanchor": "top", "x": 0,
                   "font": {"size": 12}},
        "margin": {"l": 200, "r": 30, "t": 80, "b": 170},
        "hoverlabel": {"font": {"family": _FONT_FAMILY, "size": 12}},
        "annotations": annotations,
    }
    return {"data": data, "layout": layout}


# Plotly config: horizontal-only zoom + pan. With yaxis.fixedrange set, scrollZoom
# stretches only x and pan moves only x; the vertical / box / autoscale buttons are
# stripped so the y axis can never be rescaled.
_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "zoom2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d",
    ],
}

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_HTML_TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script src="{cdn}" charset="utf-8"></script>
<style>
 html,body{{margin:0;height:100%;background:#f3ede1;font-family:ui-monospace,monospace;}}
 #tabs{{display:flex;gap:6px;padding:8px 14px 4px;overflow-x:auto;white-space:nowrap;
   border-bottom:1px solid #d8cfbb;}}
 #tabs button{{font:600 13px ui-monospace,monospace;background:#e8e1d1;color:#191a16;
   border:1px solid #c9c0b0;border-radius:5px;padding:5px 12px;cursor:pointer;}}
 #tabs button:hover{{background:#dcd3bf;}}
 #tabs button.active{{background:#7d5ba6;color:#fff;border-color:#7d5ba6;}}
 #plot{{width:100vw;height:calc(100vh - 56px);}}
</style></head><body>
<div id="tabs">{tab_buttons}</div>
<div id="plot"></div>
<script>
 const figs = {figs_json};
 const ctas = {ctas_json};
 const config = {config_json};
 const plot = document.getElementById('plot');
 function show(cta){{
   const f = figs[cta];
   Plotly.react(plot, f.data, f.layout, config);
   document.querySelectorAll('#tabs button').forEach(function(b){{
     b.classList.toggle('active', b.dataset.cta === String(cta));
   }});
 }}
 document.querySelectorAll('#tabs button').forEach(function(b){{
   b.onclick = function(){{ show(b.dataset.cta); }};
 }});
 show(ctas[0]);
 window.addEventListener('resize', function(){{ Plotly.Plots.resize(plot); }});
</script></body></html>"""


def export_timeline_html(events: list, path: str, *, group_id_to_name: dict,
                         lane_id_to_name: dict | None = None,
                         flows: list | None = None, title: str = "",
                         sm_clock_ghz: float = 1.5) -> None:
    """Write decoded trace events as a self-contained Plotly HTML timeline.

    Groups the events by CTA (``track[0]``), builds one horizontal-bar Gantt figure
    per CTA, and emits a single ``.html`` file with a top tab strip switching CTAs
    via ``Plotly.react``. Colors are assigned per event name globally (stable
    across CTAs). The viewer loads Plotly from a CDN, so opening the file needs
    internet access (or a locally cached Plotly).

    Args:
        events: Flat list of ``Slice`` / ``Instant`` objects from
            ``tileops.trace.decode.decode``.
        path: Destination ``.html`` path.
        group_id_to_name: ``gid -> name`` map (resolved from the compiled kernel),
            used to label each lane.
        lane_id_to_name: ``lane_id -> name`` map (resolved from the kernel), used
            to label each lane.
        flows: Optional declared ``(src_name, dst_name)`` pairs
            (``host_maps["flows"]``), resolved per CTA to flow arrows.
        title: Base figure title; the CTA index is appended per tab.
        sm_clock_ghz: Locked SM clock in GHz, used for the ``~ns`` hover column.

    Returns:
        None. Writes the HTML document to ``path``.
    """
    lane_id_to_name = lane_id_to_name or {}

    # All distinct event names across every CTA -> stable color assignment.
    all_names = {e.name for e in events if isinstance(e, (Slice, Instant))}
    color_map = _assign_colors(list(all_names))

    by_cta: dict = {}
    for e in events:
        by_cta.setdefault(e.track[0], []).append(e)

    ctas = sorted(by_cta)
    figs = {
        str(cta): _figure_for_cta(cta, by_cta[cta], group_id_to_name, lane_id_to_name,
                                  flows, color_map, title, sm_clock_ghz)
        for cta in ctas
    }

    tab_buttons = "".join(f'<button data-cta="{cta}">CTA {cta}</button>' for cta in ctas)
    html = _HTML_TEMPLATE.format(
        title=title or "trace timeline",
        cdn=_PLOTLY_CDN,
        tab_buttons=tab_buttons,
        figs_json=json.dumps(figs),
        ctas_json=json.dumps([str(c) for c in ctas]),
        config_json=json.dumps(_CONFIG),
    )
    with open(path, "w") as f:
        f.write(html)
