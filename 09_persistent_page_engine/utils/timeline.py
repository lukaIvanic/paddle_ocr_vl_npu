"""Low-perturbation host/device execution timeline recording.

The recorder uses the process-wide monotonic clock for host work. Device spans
are added only after the pipeline's existing event-resolution boundaries; this
module never synchronizes an accelerator.
"""

from __future__ import annotations

import json
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROW_ORDER = (
    "Pipeline",
    "Page input",
    "Layout detection",
    "Layout postprocess",
    "Crop / page preparation",
    "CPU preprocessing",
    "CPU MRoPE",
    "CPU / queue wait",
    "H2D / D2H transfer",
    "Vision prefill",
    "Text prefill",
    "Decode ready wait",
    "Decode admission",
    "Text decode",
    "Decode request residency",
    "Decode control / wait",
    "Result assembly",
    "Artifacts / tracing",
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return str(value)


class TimelineRecorder:
    """Thread-safe in-memory trace with one shared host monotonic clock."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self.reset()

    def reset(self, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.origin_ns = time.perf_counter_ns()
            self.metadata = _json_value(metadata or {})
            self._events: list[dict[str, Any]] = []
            self._next_id = 1

    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()

    def _relative_ns(self, absolute_ns: int) -> int:
        return max(0, int(absolute_ns) - int(self.origin_ns))

    def record_span(
        self,
        row: str,
        name: str,
        start_ns: int,
        end_ns: int,
        *,
        flow_id: str | None = None,
        flow_ids: list[str] | tuple[str, ...] | None = None,
        event_type: str = "work",
        clock: str = "host_monotonic",
        args: dict[str, Any] | None = None,
        thread_name: str | None = None,
    ) -> int | None:
        if not self.enabled:
            return None
        relative_start = self._relative_ns(start_ns)
        relative_end = max(relative_start, self._relative_ns(end_ns))
        related_flows = [str(item) for item in (flow_ids or ())]
        if flow_id is not None:
            match = re.search(r"page_(\d+)", str(flow_id))
            if match is not None:
                page_flow = f"page:{int(match.group(1))}"
                if page_flow not in related_flows:
                    related_flows.append(page_flow)
        with self._lock:
            event_id = self._next_id
            self._next_id += 1
            self._events.append(
                {
                    "id": event_id,
                    "kind": "span",
                    "row": str(row),
                    "name": str(name),
                    "start_ns": relative_start,
                    "end_ns": relative_end,
                    "duration_ns": relative_end - relative_start,
                    "flow_id": None if flow_id is None else str(flow_id),
                    "flow_ids": related_flows,
                    "event_type": str(event_type),
                    "clock": str(clock),
                    "thread": thread_name or threading.current_thread().name,
                    "args": _json_value(args or {}),
                }
            )
        return event_id

    def record_span_seconds(
        self,
        row: str,
        name: str,
        start_seconds: float,
        end_seconds: float,
        **kwargs: Any,
    ) -> int | None:
        return self.record_span(
            row,
            name,
            int(float(start_seconds) * 1_000_000_000),
            int(float(end_seconds) * 1_000_000_000),
            **kwargs,
        )

    @contextmanager
    def span(
        self,
        row: str,
        name: str,
        *,
        flow_id: str | None = None,
        event_type: str = "work",
        args: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record_span(
                row,
                name,
                started,
                time.perf_counter_ns(),
                flow_id=flow_id,
                event_type=event_type,
                args=args,
            )

    def instant(
        self,
        row: str,
        name: str,
        *,
        timestamp_ns: int | None = None,
        flow_id: str | None = None,
        event_type: str = "instant",
        args: dict[str, Any] | None = None,
    ) -> int | None:
        if not self.enabled:
            return None
        stamp = time.perf_counter_ns() if timestamp_ns is None else int(timestamp_ns)
        return self.record_span(
            row,
            name,
            stamp,
            stamp,
            flow_id=flow_id,
            event_type=event_type,
            args=args,
        )

    def counter(
        self,
        row: str,
        name: str,
        value: int | float,
        *,
        timestamp_ns: int | None = None,
        args: dict[str, Any] | None = None,
    ) -> int | None:
        payload = dict(args or {})
        payload["value"] = value
        return self.instant(
            row,
            name,
            timestamp_ns=timestamp_ns,
            event_type="counter",
            args=payload,
        )

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def snapshot(self) -> dict[str, Any]:
        events = self.events()
        end_ns = max((int(event["end_ns"]) for event in events), default=0)
        rows = [row for row in ROW_ORDER if any(event["row"] == row for event in events)]
        rows.extend(
            sorted(
                {str(event["row"]) for event in events}.difference(rows)
            )
        )
        return {
            "schema_version": 1,
            "clock": "nanoseconds relative to one process-wide perf_counter_ns origin",
            "device_clock_note": (
                "device-event spans use event-relative offsets anchored at host enqueue; "
                "they are resolved only at synchronization points already present in the pipeline"
            ),
            "metadata": self.metadata,
            "duration_ns": end_ns,
            "rows": rows,
            "events": sorted(events, key=lambda event: (event["start_ns"], event["id"])),
        }

    def write_json(self, path: Path) -> None:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def write_html(self, path: Path) -> None:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.snapshot(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).replace("</", "<\\/")
        path.write_text(_standalone_html(payload), encoding="utf-8")


def _standalone_html(payload: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PaddleOCR-VL pipeline timeline</title>
<style>
:root {{
  color-scheme: light dark;
  --bg:#f7f7f8;--fg:#171719;--muted:#67676d;--panel:#ffffff;--line:#d9d9de;
  --work:#3377d6;--wait:#d88722;--io:#7b5dc7;--instant:#3c9b68;--selected:#e43d68;
}}
@media (prefers-color-scheme:dark) {{ :root {{
  --bg:#151517;--fg:#ededf0;--muted:#a8a8b0;--panel:#202024;--line:#3b3b42;
  --work:#6fa8f7;--wait:#f1aa4e;--io:#ad91ef;--instant:#70c894;--selected:#ff6c91;
}} }}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,sans-serif}}
main{{padding:14px;max-width:1800px;margin:auto}} h1{{font-size:17px;font-weight:500;margin:0}}
.top{{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:10px}}
.meta,.controls,.legend{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;color:var(--muted)}}
button{{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 9px;cursor:pointer}}
button:hover{{border-color:var(--fg)}} .sw{{width:10px;height:10px;border-radius:2px;display:inline-block}}
#wrap{{position:relative;border:1px solid var(--line);background:var(--panel);border-radius:8px;overflow:hidden}}
#timeline{{display:block;width:100%;touch-action:none}} #tip{{position:absolute;display:none;pointer-events:none;max-width:420px;padding:7px 9px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg);box-shadow:0 4px 18px #0003;white-space:pre-wrap;z-index:2}}
#detail{{margin-top:9px;min-height:40px;padding:8px 10px;border-left:3px solid var(--selected);background:var(--panel);color:var(--muted)}}
#detail strong{{color:var(--fg);font-weight:500}} code{{color:var(--fg)}}
.hint{{color:var(--muted);margin-top:7px}} .metric{{font-variant-numeric:tabular-nums}}
</style>
</head>
<body><main>
<div class="top"><div><h1>PaddleOCR-VL execution timeline</h1><div class="meta" id="meta"></div></div>
<div class="controls"><button id="zoomOut" type="button">Zoom out</button><button id="reset" type="button">Reset view</button></div></div>
<div class="legend"><span class="sw" style="background:var(--work)"></span>work <span class="sw" style="background:var(--wait)"></span>wait <span class="sw" style="background:var(--io)"></span>I/O <span class="sw" style="background:var(--instant)"></span>event <span class="sw" style="background:var(--selected)"></span>selected flow</div>
<div id="wrap"><canvas id="timeline" aria-label="Execution events arranged by category on a shared time axis"></canvas><div id="tip"></div></div>
<div id="detail"><strong>Select an event</strong> to highlight that request or page across rows and expose its dependency chain.</div>
<div class="hint">Mouse wheel: zoom around pointer · drag: pan · click: follow one request/page · double-click: reset</div>
<script id="trace-data" type="application/json">{payload}</script>
<script>
(() => {{
  const trace=JSON.parse(document.getElementById('trace-data').textContent);
  const canvas=document.getElementById('timeline'),ctx=canvas.getContext('2d');
  const wrap=document.getElementById('wrap'),tip=document.getElementById('tip'),detail=document.getElementById('detail');
  const labelW=190,axisH=34,rowH=31,padR=12;
  const rows=trace.rows,events=trace.events;
  const duration=Math.max(trace.duration_ns/1e9,0.001);
  let viewStart=0,viewEnd=duration,drag=null,hover=null,selectedFlow=null,hit=[];
  const colors={{work:'--work',wait:'--wait',io:'--io',instant:'--instant',counter:'--instant'}};
  const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt=s=>s<0.001?(s*1e6).toFixed(1)+' µs':s<1?(s*1e3).toFixed(2)+' ms':s.toFixed(3)+' s';
  const rowIndex=new Map(rows.map((r,i)=>[r,i]));
  function resize(){{
    const dpr=window.devicePixelRatio||1,w=Math.max(320,wrap.clientWidth),h=axisH+rows.length*rowH+6;
    canvas.style.height=h+'px';canvas.width=Math.floor(w*dpr);canvas.height=Math.floor(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw();
  }}
  function xFor(s,w){{return labelW+(s-viewStart)/(viewEnd-viewStart)*(w-labelW-padR)}}
  function eventText(e){{
    const d=(e.end_ns-e.start_ns)/1e9,a=Object.entries(e.args||{{}}).map(([k,v])=>k+'='+JSON.stringify(v)).join(' · ');
    return e.row+'\\n'+e.name+'\\n'+fmt(d)+' @ '+(e.start_ns/1e9).toFixed(6)+' s'+(e.flow_id?'\\nflow: '+e.flow_id:'')+(a?'\\n'+a:'');
  }}
  function draw(){{
    const w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);hit=[];
    ctx.font='12px system-ui';ctx.textBaseline='middle';
    ctx.fillStyle=css('--panel');ctx.fillRect(0,0,w,h);
    ctx.strokeStyle=css('--line');ctx.lineWidth=1;
    for(let i=0;i<=rows.length;i++){{const y=axisH+i*rowH+.5;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}}
    ctx.fillStyle=css('--fg');rows.forEach((r,i)=>ctx.fillText(r,10,axisH+i*rowH+rowH/2));
    ctx.beginPath();ctx.moveTo(labelW-.5,0);ctx.lineTo(labelW-.5,h);ctx.stroke();
    const span=viewEnd-viewStart,rough=span/8,pow=Math.pow(10,Math.floor(Math.log10(rough))),step=[1,2,5,10].find(v=>v*pow>=rough)*pow;
    ctx.fillStyle=css('--muted');ctx.textAlign='center';
    for(let t=Math.ceil(viewStart/step)*step;t<=viewEnd;t+=step){{const x=xFor(t,w);ctx.beginPath();ctx.moveTo(x,axisH-7);ctx.lineTo(x,h);ctx.stroke();ctx.fillText(t.toFixed(step<1?2:1)+' s',x,13)}}
    ctx.textAlign='left';
    const visible=events.filter(e=>e.end_ns/1e9>=viewStart&&e.start_ns/1e9<=viewEnd&&rowIndex.has(e.row));
    visible.forEach(e=>{{
      const i=rowIndex.get(e.row),y=axisH+i*rowH+6,s=e.start_ns/1e9,en=e.end_ns/1e9;
      let x=xFor(s,w),x2=xFor(en,w),ew=Math.max(2,x2-x),eh=rowH-12;
      const selected=selectedFlow&&(e.flow_id===selectedFlow||(e.flow_ids||[]).includes(selectedFlow));
      ctx.fillStyle=selected?css('--selected'):css(colors[e.event_type]||'--work');
      if(e.kind==='instant'||e.duration_ns===0){{ctx.beginPath();ctx.moveTo(x,y+eh/2-5);ctx.lineTo(x+5,y+eh/2);ctx.lineTo(x,y+eh/2+5);ctx.lineTo(x-5,y+eh/2);ctx.closePath();ctx.fill();hit.push({{e,x:x-6,y,w:12,h:eh}})}}
      else{{ctx.globalAlpha=e.clock==='device_event_reconstructed'?0.92:0.78;ctx.fillRect(x,y,ew,eh);ctx.globalAlpha=1;if(e.event_type==='wait'){{ctx.strokeStyle=css('--panel');for(let sx=x-eh;sx<x+ew;sx+=6){{ctx.beginPath();ctx.moveTo(sx,y+eh);ctx.lineTo(sx+eh,y);ctx.stroke()}}ctx.strokeStyle=css('--line')}}hit.push({{e,x,y,w:ew,h:eh}})}}
      if(ew>46){{ctx.save();ctx.beginPath();ctx.rect(x+2,y,ew-4,eh);ctx.clip();ctx.fillStyle=css('--fg');ctx.fillText(e.name,x+4,y+eh/2);ctx.restore()}}
    }});
    if(selectedFlow){{
      const chain=events.filter(e=>(e.flow_id===selectedFlow||(e.flow_ids||[]).includes(selectedFlow))&&rowIndex.has(e.row)).sort((a,b)=>a.start_ns-b.start_ns);
      ctx.strokeStyle=css('--selected');ctx.lineWidth=1.4;ctx.globalAlpha=.75;
      let prior=null;
      for(const current of chain){{if(prior&&current.start_ns>=prior.end_ns){{const ax=xFor(prior.end_ns/1e9,w),ay=axisH+rowIndex.get(prior.row)*rowH+rowH/2,bx=xFor(current.start_ns/1e9,w),by=axisH+rowIndex.get(current.row)*rowH+rowH/2;if(ax>=labelW&&ax<=w&&bx>=labelW&&bx<=w){{ctx.beginPath();ctx.moveTo(ax,ay);ctx.bezierCurveTo((ax+bx)/2,ay,(ax+bx)/2,by,bx,by);ctx.stroke()}}}}if(prior===null||current.end_ns>prior.end_ns)prior=current}}
      ctx.globalAlpha=1;ctx.lineWidth=1;
    }}
  }}
  function point(ev){{const r=canvas.getBoundingClientRect();return{{x:ev.clientX-r.left,y:ev.clientY-r.top}}}}
  function findHit(p){{for(let i=hit.length-1;i>=0;i--){{const q=hit[i];if(p.x>=q.x&&p.x<=q.x+q.w&&p.y>=q.y&&p.y<=q.y+q.h)return q.e}}return null}}
  canvas.addEventListener('mousemove',ev=>{{const p=point(ev);if(drag){{const dx=p.x-drag.x,sec=dx/(canvas.clientWidth-labelW-padR)*(viewEnd-viewStart);viewStart=Math.max(0,Math.min(duration-(viewEnd-viewStart),drag.start-sec));viewEnd=viewStart+drag.span;draw();return}}hover=findHit(p);if(hover){{tip.style.display='block';tip.textContent=eventText(hover);const tw=tip.offsetWidth,th=tip.offsetHeight;tip.style.left=Math.max(4,Math.min(canvas.clientWidth-tw-4,p.x+12))+'px';tip.style.top=Math.max(4,Math.min(canvas.clientHeight-th-4,p.y+12))+'px'}}else tip.style.display='none'}});
  canvas.addEventListener('mousedown',ev=>{{const p=point(ev);drag={{x:p.x,start:viewStart,span:viewEnd-viewStart}}}});
  window.addEventListener('mouseup',()=>drag=null);
  canvas.addEventListener('mouseleave',()=>{{drag=null;tip.style.display='none'}});
  canvas.addEventListener('wheel',ev=>{{ev.preventDefault();const p=point(ev),ratio=Math.max(0,Math.min(1,(p.x-labelW)/(canvas.clientWidth-labelW-padR))),factor=ev.deltaY>0?1.45:.69,old=viewEnd-viewStart,next=Math.max(duration/100000,Math.min(duration,old*factor)),anchor=viewStart+ratio*old;viewStart=Math.max(0,Math.min(duration-next,anchor-ratio*next));viewEnd=viewStart+next;draw()}},{{passive:false}});
  canvas.addEventListener('click',ev=>{{const e=findHit(point(ev));selectedFlow=e&&e.flow_id?e.flow_id:null;if(e)detail.innerHTML='<strong>'+escapeHtml(e.name)+'</strong> · '+escapeHtml(e.row)+' · <span class="metric">'+fmt(e.duration_ns/1e9)+'</span><br>'+escapeHtml(eventText(e).split('\\n').slice(3).join(' · '));else detail.innerHTML='<strong>Select an event</strong> to highlight that request or page across rows and expose its dependency chain.';draw()}});
  canvas.addEventListener('dblclick',reset);
  function escapeHtml(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
  function reset(){{viewStart=0;viewEnd=duration;selectedFlow=null;draw()}}
  document.getElementById('reset').onclick=reset;
  document.getElementById('zoomOut').onclick=()=>{{const mid=(viewStart+viewEnd)/2,next=Math.min(duration,(viewEnd-viewStart)*2);viewStart=Math.max(0,Math.min(duration-next,mid-next/2));viewEnd=viewStart+next;draw()}};
  document.getElementById('meta').innerHTML='<span class="metric">'+duration.toFixed(3)+' s</span><span>'+events.length.toLocaleString()+' events</span><span>'+rows.length+' rows</span>';
  new ResizeObserver(resize).observe(wrap);resize();
}})();
</script>
</main></body></html>
"""
