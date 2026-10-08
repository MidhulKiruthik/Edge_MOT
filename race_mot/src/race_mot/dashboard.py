"""Small local dashboard for inspecting a completed or running replay.

The dashboard serves run metadata and temporary track geometry only. It never
serves source frames, credentials, or raw video. Bind to loopback by default;
an operator may explicitly select a trusted-LAN address when needed.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _last_frame(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    last: dict[str, Any] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        last = value
    except (OSError, json.JSONDecodeError):
        return last
    return last


def load_dashboard_state(run_dir: Path) -> dict[str, Any]:
    """Return a redacted, JSON-safe snapshot for the browser dashboard."""
    manifest = _read_json(run_dir / "manifest.json")
    summary = _read_json(run_dir / "summary.json")
    frame = _last_frame(run_dir / "frame_log.jsonl")
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    dimensions = runtime.get("sequence_dimensions")
    if not (isinstance(dimensions, list) and len(dimensions) == 2):
        dimensions = [1280, 720]
    tracks = []
    for item in frame.get("tracks", []):
        if not isinstance(item, dict):
            continue
        tracks.append(
            {
                "temporary_track_id": item.get("temporary_track_id"),
                "xyxy": item.get("xyxy"),
                "score": item.get("score"),
                "state": item.get("state"),
            }
        )
    return {
        "schema_version": 1,
        "run_id": summary.get("run_id") or manifest.get("run_id"),
        "status": summary.get("status", "running"),
        "source": manifest.get("source") or manifest.get("source_name"),
        "source_kind": manifest.get("source_kind"),
        "phone_capture": False,
        "frames_processed": summary.get("frames_processed", 0),
        "detections": summary.get("detections", 0),
        "tracks_emitted": summary.get("tracks", 0),
        "error": summary.get("error"),
        "runtime": {
            "action": runtime.get("action", "DETECT"),
            "max_queue_size": runtime.get("max_queue_size"),
            "deadline_ms": runtime.get("deadline_ms"),
        },
        "dimensions": dimensions,
        "latest": {
            "source_index": frame.get("source_index"),
            "source_timestamp_ms": frame.get("source_timestamp_ms"),
            "planned_action": frame.get("planned_action"),
            "executed_action": frame.get("executed_action"),
            "reason": frame.get("reason"),
            "pipeline_latency_ms": frame.get("pipeline_latency_ms"),
            "processing_ms": frame.get("processing_ms"),
            "active_track_count": frame.get("active_track_count", len(tracks)),
            "queue_depth": frame.get("queue_depth", 0),
            "input_drop_count": frame.get("input_drop_count", 0),
            "tracks": tracks,
        },
    }


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RACE-MOT local replay</title>
<style>
body{font:15px system-ui,sans-serif;background:#111827;color:#e5e7eb;margin:0;padding:24px}
main{max-width:1100px;margin:auto}h1{font-size:24px;margin:0 0 4px}small{color:#9ca3af}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}
.card{background:#1f2937;border:1px solid #374151;border-radius:8px;padding:14px}.label{color:#9ca3af;font-size:12px;text-transform:uppercase}.value{font-size:22px;margin-top:5px}
#error{color:#fca5a5;min-height:1.4em}.scene{background:#0b1220;border:1px solid #374151;border-radius:8px;padding:12px}
svg{width:100%;height:420px;background:#0f172a}.box{fill:none;stroke:#22d3ee;stroke-width:2}.tag{fill:#22d3ee;font:12px system-ui}.muted{color:#9ca3af}
</style></head><body><main>
<h1>RACE-MOT local replay dashboard</h1><small id="run">Loading redacted run state…</small>
<div class="grid"><div class="card"><div class="label">Run status</div><div id="status" class="value">—</div></div>
<div class="card"><div class="label">Source frame</div><div id="frame" class="value">—</div></div>
<div class="card"><div class="label">Active temporary tracks</div><div id="tracks" class="value">—</div></div>
<div class="card"><div class="label">Pipeline latency</div><div id="latency" class="value">—</div></div>
<div class="card"><div class="label">Action / reason</div><div id="action" class="value">—</div></div>
<div class="card"><div class="label">Queue / drops</div><div id="queue" class="value">—</div></div></div>
<div id="error"></div><div class="scene"><div class="muted">Temporary track geometry (source frames are never served)</div><svg id="overlay" viewBox="0 0 1280 720" aria-label="temporary track overlay"></svg></div>
</main><script>
const text=(id,value)=>document.getElementById(id).textContent=value ?? '—';
function draw(tracks){const svg=document.getElementById('overlay'); while(svg.firstChild)svg.removeChild(svg.firstChild);
 for(const t of (tracks||[])){const b=t.xyxy||[];if(b.length!==4)continue;const [x1,y1,x2,y2]=b;
  const r=document.createElementNS('http://www.w3.org/2000/svg','rect');r.setAttribute('class','box');r.setAttribute('x',x1);r.setAttribute('y',y1);r.setAttribute('width',Math.max(0,x2-x1));r.setAttribute('height',Math.max(0,y2-y1));svg.appendChild(r);
  const label=document.createElementNS('http://www.w3.org/2000/svg','text');label.setAttribute('class','tag');label.setAttribute('x',x1);label.setAttribute('y',Math.max(14,y1-4));label.textContent='ID '+t.temporary_track_id;svg.appendChild(label);
 }}
async function refresh(){try{const r=await fetch('/api/state',{cache:'no-store'});const d=await r.json();const l=d.latest||{};
 document.getElementById('overlay').setAttribute('viewBox','0 0 '+(d.dimensions?.[0]||1280)+' '+(d.dimensions?.[1]||720));
 text('run',(d.source||'local replay')+' · '+(d.source_kind||'')+' · run '+(d.run_id||'—'));text('status',d.status);text('frame',l.source_index);
 text('tracks',l.active_track_count);text('latency',l.pipeline_latency_ms==null?'—':Number(l.pipeline_latency_ms).toFixed(2)+' ms');
 text('action',(l.executed_action||d.runtime?.action||'—')+' / '+(l.reason||'—'));text('queue',(l.queue_depth??0)+' / '+(l.input_drop_count??0));
 text('error',d.error||'');draw(l.tracks);}catch(e){text('error','Dashboard state unavailable: '+e);}}
refresh();setInterval(refresh,1000);
</script></body></html>"""


def serve_dashboard(run_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve a local dashboard until interrupted."""
    run_dir = run_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlsplit(self.path).path
            if path == "/api/state":
                body = json.dumps(load_dashboard_state(run_dir), separators=(",", ":")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
            elif path in {"/", "/index.html"}:
                body = _HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
            else:
                body = b"Not found\n"
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"RACE-MOT dashboard: http://{host}:{server.server_address[1]}/")
    print("Only redacted run metadata and temporary track geometry are served.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
