from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from routing import Router

app = FastAPI(title="Kestrel Home Service Request Router", version="1.0")

try:
    router = Router()
    load_error = None
except Exception as exc:
    router = None
    load_error = str(exc)


class ServiceRequest(BaseModel):
    request_id: Optional[str] = None
    created_at_ist: Optional[str] = None
    channel: Optional[str] = None
    product_family: Optional[str] = None
    warranty_status: Optional[str] = None
    request_text: str = Field(min_length=1)
    source: Optional[str] = None


@app.post("/route")
def route(req: ServiceRequest):
    if router is None:
        raise HTTPException(status_code=503, detail=f"Routing model is not ready: {load_error}")
    try:
        output = router.predict(req.model_dump())
        output["request_id"] = req.request_id
        return output
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not route request: {exc}") from exc


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def screen() -> str:
    return r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kestrel Home - Service Request Router</title>
<style>
body{font-family:Arial,sans-serif;background:#f4f6f8;margin:0;color:#1e293b}.wrap{max-width:900px;margin:36px auto;padding:0 20px}.card{background:white;border-radius:14px;box-shadow:0 3px 18px rgba(0,0,0,.08);padding:26px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}label{font-weight:600;font-size:14px;display:block;margin-bottom:6px}input,select,textarea{width:100%;box-sizing:border-box;padding:10px;border:1px solid #cbd5e1;border-radius:8px;font:inherit}textarea{min-height:130px;resize:vertical}.full{grid-column:1/-1}button{margin-top:18px;padding:11px 18px;border:0;border-radius:8px;font-weight:700;cursor:pointer;background:#0f172a;color:#fff}.result{margin-top:22px;padding:18px;border-radius:10px;background:#f8fafc;border:1px solid #e2e8f0}.team{font-size:24px;font-weight:800}.reason{margin:9px 0}.muted{color:#64748b;font-size:13px}.pill{display:inline-block;margin-left:8px;padding:3px 8px;border-radius:999px;background:#e2e8f0;font-size:12px}.notes{margin-top:18px;font-size:13px;color:#475569}</style>
</head><body><div class="wrap"><div class="card">
<h1>Kestrel Home - Service Request Router</h1><div class="muted">Local policy-aware classifier. No paid API key required.</div>
<form id="f"><div class="grid">
<div><label>Request ID</label><input id="request_id" value="SR-DEMO-001"></div>
<div><label>Channel</label><select id="channel"><option>chat</option><option>whatsapp</option><option>ivr</option><option>email</option></select></div>
<div><label>Product</label><select id="product_family"><option>Air Fryer</option><option>Mixer Grinder</option><option>Water Purifier</option><option>Robot Vacuum</option><option>Induction Cooktop</option><option>Ceiling Fan</option><option>Room Heater</option></select></div>
<div><label>Warranty status</label><select id="warranty_status"><option>in_warranty</option><option>out_of_warranty</option><option>shield</option></select></div>
<div class="full"><label>Customer request</label><textarea id="request_text">The purifier display is blank and it is not turning on.</textarea></div>
<div><label>Source</label><select id="source"><option>crm</option><option>legacy_zoho</option></select></div>
</div><button type="submit">Route request</button></form>
<div id="result"></div>
<div class="notes"><b>Build notes:</b> word + character TF-IDF + LinearSVC. Keyword-only routing was discarded for lower validation accuracy; a paid LLM/API was not needed.</div>
</div></div>
<script>
const form=document.getElementById('f'), result=document.getElementById('result');
form.addEventListener('submit', async (e)=>{e.preventDefault(); result.innerHTML='<div class="result">Routing...</div>';
const payload={request_id:request_id.value,channel:channel.value,product_family:product_family.value,warranty_status:warranty_status.value,request_text:request_text.value,source:source.value};
try{const r=await fetch('/route',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); const j=await r.json(); if(!r.ok) throw new Error(j.detail||'Request failed');
result.innerHTML='<div class="result"><div class="team">'+j.team+' <span class="pill">'+Math.round(j.confidence*100)+'% relative confidence</span></div><h3>Why</h3>'+j.reasons.map(x=>'<div class="reason">- '+x+'</div>').join('')+'</div>';
}catch(err){result.innerHTML='<div class="result"><b>Could not route:</b> '+err.message+'</div>';}});
</script></body></html>'''
