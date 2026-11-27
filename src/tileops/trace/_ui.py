"""Self-contained Plotly HTML timeline of decoded trace events, one tab per CTA.

Each CTA's figure has one categorical lane per ``(gid, lane)`` pair that carries events,
labelled ``"<group> / <lane>"`` with the first pair at the top; one bar trace per event
name, colored by sorted name so a name keeps its color across CTAs; and one annotation
arrow per ``FlowEdge``. Zoom and pan move the x axis only. The page loads Plotly from a
CDN; this module imports neither torch nor plotly.
"""

import json

from tileops.trace._decode import Instant, Slice, compute_flows

__all__ = ["export_timeline_html"]

# Event names take these colors in sorted order, so a name keeps its color across CTAs.
_PALETTE = [
    "#c43c19",  # red-orange
    "#2f5d8c",  # steel blue
    "#1a6e63",  # teal
    "#d8a13a",  # amber
    "#7d5ba6",  # violet
    "#7f97a6",  # slate
    "#9a8f7a",  # taupe
    "#5f6d76",  # gunmetal
    "#b5651d",  # ochre
    "#4a7c59",  # moss
    "#8c4a6b",  # plum
    "#3d6b8c",  # denim
    "#a6803a",  # bronze
    "#6b6f5f",  # olive grey
    "#9c5d3a",  # terracotta
    "#5a6b8c",  # dusty blue
]

_ARROW_COLOR = "#7d5ba6"

_PAPER_BG = "#f3ede1"
_GRID_COLOR = "#d8cfbb"
_FONT_FAMILY = "ui-monospace, SFMono-Regular, monospace"
_INK = "#191a16"


# With yaxis.fixedrange set, scroll-zoom and pan move only x; the buttons that could
# rescale y are removed.
_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
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


def _lane_label(gid: int, lane: int, group_id_to_name: dict, lane_id_to_name: dict) -> str:
    """A lane's y-axis label ``"<group> / <lane>"``."""
    gname = group_id_to_name.get(gid, f"g{gid}")
    lname = lane_id_to_name.get(lane, str(lane))
    return f"{gname} / {lname}"


def _text_color(hex_color: str) -> str:
    """Ink on a light ``#rrggbb`` fill, white on a dark one."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return _INK if luminance > 0.6 else "#ffffff"


def _bar_trace(name, color, lane_labels, bases, durs, texts, payloads, sm_clock_ghz, *, width):
    """One Plotly horizontal-bar trace: one bar per event of *name*."""
    # cycles / GHz == ns at the locked SM clock.
    customdata = [
        [name, "" if p is None else p, d, round(d / sm_clock_ghz, 1)]
        for d, p in zip(durs, payloads, strict=True)
    ]
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
        "textangle": 0,
        "textfont": {"color": _text_color(color), "size": 11, "family": _FONT_FAMILY},
        "marker": {"color": color, "line": {"color": _PAPER_BG, "width": 1.2}},
        "width": width,
        "showlegend": True,
        "customdata": customdata,
        "hovertemplate": (
            "<b>%{customdata[0]}</b>  payload %{customdata[1]}<br>"
            "start %{base} cyc   dur %{x} cyc "
            "(~%{customdata[3]} ns)<extra></extra>"
        ),
    }


def _figure_for_cta(
    cta, events, group_id_to_name, lane_id_to_name, flows, color_map, title, sm_clock_ghz
):
    """The Plotly ``{data, layout}`` figure of one CTA's events."""
    slices = [e for e in events if isinstance(e, Slice)]
    instants = [e for e in events if isinstance(e, Instant)]
    # compute_flows pairs same-named slices in timestamp order; events are already
    # restricted to this CTA, so each declared flow yields its per-occurrence edges.
    edges = compute_flows(events, flows or [])

    # Lanes auto-derived: distinct (gid, lane) with events, sorted by (gid, lane).
    lane_pairs = sorted({(e.track[1], e.track[2]) for e in events})
    lane_labels = {
        (g, l): _lane_label(g, l, group_id_to_name, lane_id_to_name) for (g, l) in lane_pairs
    }

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
    # Slices by name, then instants by name as thin bars of 1% of the span.
    for kind, width in ((Slice, 0.62), (Instant, 0.34)):
        by_name: dict = {}
        for e in events:
            if isinstance(e, kind):
                by_name.setdefault(e.name, []).append(e)
        for name in sorted(by_name):
            items = by_name[name]
            if kind is Slice:
                durs = [s.dur_cy for s in items]
                text = [name if s.payload is None else f"{name}<br>{s.payload}" for s in items]
            else:
                durs = [instant_w] * len(items)
                text = [name] * len(items)
            data.append(
                _bar_trace(
                    name,
                    color_map[name],
                    [lane_labels[(e.track[1], e.track[2])] for e in items],
                    [e.ts_cy for e in items],
                    durs,
                    text,
                    [e.payload for e in items],
                    sm_clock_ghz,
                    width=width,
                )
            )

    # Plotly draws categoryarray bottom-up; reverse the sorted lane order so the
    # first (gid, lane) — producer / group0 main — renders at the TOP.
    category_array = [lane_labels[p] for p in reversed(lane_pairs)]

    annotations = []
    for edge in edges:
        annotations.append(
            {
                "x": edge.dst_ts_cy,
                "y": _lane_label(*edge.dst_track[1:], group_id_to_name, lane_id_to_name),
                "ax": edge.src_ts_cy,
                "ay": _lane_label(*edge.src_track[1:], group_id_to_name, lane_id_to_name),
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 2,
                "arrowwidth": 1.5,
                "arrowsize": 1.1,
                "arrowcolor": _ARROW_COLOR,
                "opacity": 0.85,
                "text": "",
            }
        )

    layout = {
        "title": {
            "text": f"<b>{title} · CTA {cta}</b>",
            "font": {"size": 20, "family": _FONT_FAMILY},
            "x": 0.02,
            "xanchor": "left",
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
        "legend": {"orientation": "h", "y": -0.30, "yanchor": "top", "x": 0, "font": {"size": 12}},
        "margin": {"l": 200, "r": 30, "t": 80, "b": 170},
        "hoverlabel": {"font": {"family": _FONT_FAMILY, "size": 12}},
        "annotations": annotations,
    }
    return {"data": data, "layout": layout}


def export_timeline_html(
    events: list,
    path: str,
    *,
    group_id_to_name: dict,
    lane_id_to_name: dict | None = None,
    flows: list | None = None,
    title: str = "",
    sm_clock_ghz: float = 1.5,
) -> None:
    """Write decoded trace events as a self-contained Plotly HTML timeline.

    Args:
        events: ``Slice`` / ``Instant`` objects from ``decode``.
        path: Destination ``.html`` path.
        group_id_to_name: ``gid -> name`` map, for lane labels.
        lane_id_to_name: ``lane_id -> name`` map, for lane labels.
        flows: Declared ``(src_name, dst_name)`` pairs, drawn as arrows per CTA.
        title: Base figure title; the CTA index is appended per tab.
        sm_clock_ghz: Locked SM clock in GHz, for the ``~ns`` hover column.
    """
    lane_id_to_name = lane_id_to_name or {}

    # All distinct event names across every CTA -> stable color assignment.
    all_names = {e.name for e in events if isinstance(e, (Slice, Instant))}
    color_map = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(sorted(all_names))}

    by_cta: dict = {}
    for e in events:
        by_cta.setdefault(e.track[0], []).append(e)

    ctas = sorted(by_cta)
    figs = {
        str(cta): _figure_for_cta(
            cta,
            by_cta[cta],
            group_id_to_name,
            lane_id_to_name,
            flows,
            color_map,
            title,
            sm_clock_ghz,
        )
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
