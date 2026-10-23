
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#090b10">
<title>Medical Emergency | Kerala Flood AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
 --bg:#090b10;--sidebar:#0d1017;--panel:#151922;--panel2:#1b202b;
 --gold:#d8b671;--gold2:#f5daa0;--text:#f4f1e9;--muted:#9399a7;
 --line:rgba(255,255,255,.085);--green:#54d6a4;--orange:#ffb66e;
 --red:#ff777d;--blue:#82adff;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font:13px "DM Sans",sans-serif}
button,input,select,textarea{font:inherit}
button{cursor:pointer}
a{text-decoration:none;color:inherit}
:focus-visible{outline:2px solid var(--gold);outline-offset:3px}
.app{display:grid;grid-template-columns:252px minmax(0,1fr);min-height:100vh}
.sidebar{position:sticky;top:0;height:100vh;background:var(--sidebar);border-right:1px solid var(--line);display:flex;flex-direction:column;z-index:40}
.brand{padding:24px 18px;display:flex;gap:12px;align-items:center;border-bottom:1px solid var(--line)}
.brand-icon{width:43px;height:43px;display:grid;place-items:center;border-radius:13px;background:linear-gradient(135deg,#f5daa0,#a47c3b);color:#20170a;font-size:23px}
.brand strong{font:800 13px Manrope,sans-serif;letter-spacing:.4px}
.brand small{display:block;font-size:10px;letter-spacing:1.2px;color:var(--muted);margin-top:5px}
.nav{padding:18px 13px;overflow:auto;flex:1}
.nav-label{color:#777e8d;font-size:10px;letter-spacing:1.5px;font-weight:800;margin:17px 12px 9px}
.nav a{display:flex;align-items:center;gap:12px;padding:12px 13px;margin-bottom:5px;border:1px solid transparent;border-radius:10px;color:#a2a8b4;font-size:12px;font-weight:600;transition:.2s}
.nav a:hover{background:#1b202b;color:white}
.nav a.active{background:#d8b67119;border-color:#d8b67130;color:var(--gold2)}
.nav-icon{width:20px;text-align:center;font-size:17px}
.side-footer{padding:17px;border-top:1px solid var(--line);display:flex;align-items:center;gap:10px}
.avatar{width:38px;height:38px;border-radius:11px;background:#383023;color:var(--gold2);display:grid;place-items:center;font-weight:800}
.side-footer strong{font-size:11px}
.side-footer small{display:block;color:var(--muted);font-size:10px;margin-top:5px}
.main{min-width:0;background:radial-gradient(ellipse at 65% -5%,#d8b67112,transparent 38%),var(--bg)}
.topbar{min-height:82px;padding:17px 30px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:14px}
.top-left,.top-actions{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.page-icon{width:43px;height:43px;border-radius:12px;background:#d8b67119;border:1px solid #d8b67133;color:var(--gold);display:grid;place-items:center;font-size:20px}
.top-left h1{font:800 18px Manrope,sans-serif;letter-spacing:-.5px;margin:0}
.top-left p{color:var(--muted);font-size:11px;margin:5px 0 0}
.badge,.tag{display:inline-flex;align-items:center;gap:7px;padding:9px 11px;background:#d8b67112;border:1px solid #d8b67130;color:var(--gold2);border-radius:9px;font-size:10px;font-weight:800}
.dot{width:7px;height:7px;border-radius:50%;background:var(--gold);animation:pulse 2s infinite}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:11px 15px;border-radius:10px;border:1px solid var(--line);background:#1a1f29;color:#e1e3e8;font-size:11px;font-weight:800;transition:.2s}
.btn:hover{border-color:#d8b67166;transform:translateY(-1px)}
.btn.gold{background:linear-gradient(120deg,#c6a15e,#f1d49a);color:#1b160e;border:0}
.btn.red{color:var(--red);background:#ff777d12;border-color:#ff777d40}
.btn.blue{color:var(--blue);background:#82adff12;border-color:#82adff40}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
.content{max-width:1700px;margin:auto;padding:28px 30px 45px}
.eyebrow{color:var(--gold);font-size:10px;letter-spacing:2px;font-weight:800;margin-bottom:10px}
.heading{display:flex;justify-content:space-between;align-items:flex-end;gap:15px;flex-wrap:wrap;margin-bottom:22px}
.heading h2{font:800 28px Manrope,sans-serif;letter-spacing:-1px;margin:0}
.heading p{color:var(--muted);font-size:12px;line-height:1.8;margin:9px 0 0}
.hero{padding:28px;border-radius:18px;border:1px solid #d8b67138;background:radial-gradient(circle at 85% 15%,#d8b67120,transparent 35%),linear-gradient(120deg,#252017,#151922)}
.hero h3{font:800 26px Manrope,sans-serif;letter-spacing:-.8px;margin:12px 0}
.hero p{color:#b9b5ae;font-size:12px;line-height:1.9;max-width:870px}
.hero-actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}
.grid{display:grid;gap:16px}
.kpis{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:17px}
.card{min-width:0;background:linear-gradient(150deg,#181c26,#12161e);border:1px solid var(--line);border-radius:16px;box-shadow:0 14px 35px #0002;animation:rise .5s ease both}
.kpi{padding:21px}
.kpi-head{display:flex;justify-content:space-between;align-items:center;color:#a7adba;font-size:11px;font-weight:700}
.kpi-icon{width:36px;height:36px;border-radius:10px;background:#d8b67119;color:var(--gold);display:grid;place-items:center;font-size:18px}
.kpi-number{font:800 30px Manrope,sans-serif;letter-spacing:-1px;margin:15px 0 7px}
.kpi-caption{font-size:10px;color:var(--muted);line-height:1.7}
.kpi-caption b{color:var(--gold)}
.section{margin-top:25px}
.section-head{display:flex;justify-content:space-between;align-items:flex-end;gap:14px;flex-wrap:wrap;margin-bottom:15px}
.section-head h3{font:800 19px Manrope,sans-serif;letter-spacing:-.5px;margin:0}
.section-head p{color:var(--muted);font-size:11px;line-height:1.8;margin:7px 0 0}
.panel{padding:21px}
.panel-title{font:800 14px Manrope,sans-serif}
.panel-sub{color:var(--muted);font-size:10px;line-height:1.8;margin-top:6px}
.columns{grid-template-columns:minmax(0,1.45fr) minmax(310px,1fr)}
.two-col{grid-template-columns:repeat(2,minmax(0,1fr))}
.field{display:grid;gap:8px;margin-bottom:14px}
.field label{font-size:11px;font-weight:800;color:#c9ccd4}
.input,.select,.textarea{width:100%;padding:12px 13px;border:1px solid #343a46;border-radius:10px;background:#1b202b;color:var(--text);outline:none}
.input:focus,.select:focus,.textarea:focus{border-color:var(--gold)}
.textarea{resize:vertical;min-height:85px}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:0 13px}
.form-grid .wide{grid-column:1/-1}
.checkline{display:flex;align-items:flex-start;gap:10px;color:#b9bec9;font-size:11px;line-height:1.7;margin:9px 0 16px}
.checkline input{accent-color:var(--gold);margin-top:3px}
.info-list{display:grid;gap:10px}
.info-row{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:13px;background:#1b202b;border:1px solid var(--line);border-radius:10px}
.info-row strong{font-size:11px}
.info-row small{display:block;color:var(--muted);font-size:10px;margin-top:5px;line-height:1.6}
.pill{display:inline-flex;align-items:center;padding:7px 10px;border-radius:8px;font-size:10px;font-weight:800;white-space:nowrap}
.pill.critical,.pill.rejected{color:var(--red);background:#ff777d19}
.pill.high,.pill.pending{color:var(--orange);background:#ffb66e19}
.pill.medium,.pill.reviewed{color:var(--blue);background:#82adff19}
.pill.low,.pill.closed,.pill.approved{color:var(--green);background:#54d6a419}
.table-wrap{overflow:auto}
table{width:100%;border-collapse:collapse;white-space:nowrap;text-align:left}
th{padding:13px 10px;border-bottom:1px solid var(--line);color:#8d94a2;font-size:10px;letter-spacing:.5px}
td{padding:14px 10px;border-bottom:1px solid #ffffff0b;color:#d7dbe2;font-size:11px}
tbody tr:hover{background:#d8b67108}
.small-btn{padding:8px 10px;border-radius:8px;border:1px solid #d8b67140;background:#d8b67115;color:var(--gold2);font-size:10px;font-weight:800}
.filters{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:16px}
.filters .input{flex:1;min-width:180px}
.filters .select{width:auto;min-width:130px}
.resource{padding:15px;background:#1b202b;border:1px solid var(--line);border-radius:11px;margin-bottom:10px}
.resource-head{display:flex;justify-content:space-between;align-items:center;gap:10px}
.resource strong{font-size:11px}
.resource p{font-size:10px;color:var(--muted);line-height:1.8;margin:9px 0}
.progress{height:8px;background:#303642;border-radius:10px;overflow:hidden;margin-top:10px}
.progress span{display:block;height:100%;background:linear-gradient(90deg,#a68145,#f1d69c);border-radius:10px}
.timeline{display:grid;gap:15px}
.event{display:flex;gap:12px}
.event-icon{width:33px;height:33px;flex-shrink:0;display:grid;place-items:center;border-radius:10px;background:#d8b67118;color:var(--gold)}
.event strong{font-size:11px}
.event p{font-size:10px;color:var(--muted);line-height:1.7;margin:7px 0}
.event small{font-size:10px;color:#777e8d}
.notice{padding:16px;background:#d8b6710e;border:1px solid #d8b67130;border-radius:12px;color:#bbb4a5;font-size:11px;line-height:1.9;margin-top:17px}
.notice strong{color:var(--gold2)}
.footer{padding-top:26px;text-align:center;color:#69707e;font-size:10px}
.overlay{position:fixed;inset:0;z-index:100;background:#000b;display:none;align-items:center;justify-content:center;padding:18px}
.overlay.open{display:flex}
.modal{width:min(570px,100%);max-height:90vh;overflow:auto;background:#181d27;border:1px solid #d8b67144;border-radius:17px;padding:24px;box-shadow:0 30px 90px #0009}
.modal-head{display:flex;justify-content:space-between;align-items:center;gap:12px}
.modal-head h3{font:800 18px Manrope,sans-serif;margin:0}
.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:20px 0}
.detail{padding:13px;background:#10151e;border:1px solid var(--line);border-radius:9px}
.detail small{display:block;color:var(--muted);font-size:10px;margin-bottom:8px}
.detail strong{font-size:11px;overflow-wrap:anywhere}
.modal-actions{display:flex;gap:9px;flex-wrap:wrap;margin-top:20px}
.toast{position:fixed;right:20px;bottom:20px;z-index:150;padding:14px 17px;background:#252b35;border:1px solid #d8b67166;color:var(--gold2);border-radius:11px;box-shadow:0 15px 45px #0008}
.menu-btn{display:none}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{50%{opacity:.35}}
@media(max-width:1250px){.kpis{grid-template-columns:repeat(2,1fr)}.columns{grid-template-columns:1fr}}
@media(max-width:900px){.app{grid-template-columns:1fr}.sidebar{position:fixed;width:260px;transform:translateX(-100%);transition:.3s}.sidebar.open{transform:translateX(0)}.menu-btn{display:inline-flex}.topbar{padding:15px 20px}.content{padding:20px}}
@media(max-width:560px){.kpis,.two-col,.form-grid{grid-template-columns:1fr}.form-grid .wide{grid-column:auto}.heading h2{font-size:23px}.panel{padding:16px}.top-actions .badge{display:none}.detail-grid{grid-template-columns:1fr}}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation-duration:.01ms!important;transition-duration:.01ms!important}}
</style>
</head>

<body>
<div class="app">

<aside class="sidebar" id="sidebar">
 <div class="brand">
  <div class="brand-icon">✦</div>
  <div><strong>KERALA FLOOD AI</strong><small>INTELLIGENCE PLATFORM</small></div>
 </div>
 <nav class="nav">
  <div class="nav-label">WORKSPACE</div>
  <a href="index.html"><span class="nav-icon">▦</span>Project Overview</a>
  <a href="command-center.html"><span class="nav-icon">◈</span>Command Center</a>
  <a href="agent-command-center.html"><span class="nav-icon">✦</span>Agent Command Center</a>
  <a href="ai-copilot.html"><span class="nav-icon">◇</span>AI Copilot</a>
  <a href="analytics.html"><span class="nav-icon">▥</span>Analytics</a>

  <div class="nav-label">EMERGENCY RESPONSE</div>
  <a href="incident-management.html"><span class="nav-icon">◉</span>Incident Management</a>
  <a href="emergency-sos.html"><span class="nav-icon">⚑</span>Emergency SOS</a>
  <a href="rescue-dashboard.html"><span class="nav-icon">✧</span>Rescue Operations</a>
  <a href="evacuation-planning.html"><span class="nav-icon">⌁</span>Evacuation Planning</a>
  <a class="active" href="medical-emergency.html"><span class="nav-icon">✚</span>Medical Emergency</a>
  <a href="ambulance-dispatch.html"><span class="nav-icon">▣</span>Ambulance Dispatch</a>
  <a href="hospital-capacity.html"><span class="nav-icon">⌂</span>Hospital Capacity</a>

  <div class="nav-label">FLOOD INTELLIGENCE</div>
  <a href="live-flood-map.html"><span class="nav-icon">◎</span>Live Flood Map</a>
  <a href="district-dashboard.html"><span class="nav-icon">▦</span>District Dashboard</a>
  <a href="flood-monitoring.html"><span class="nav-icon">≈</span>Flood Monitoring</a>
  <a href="flood-alerts.html"><span class="nav-icon">☁</span>Flood Alerts</a>
  <a href="flood-prediction.html"><span class="nav-icon">◈</span>Flood Prediction</a>

  <div class="nav-label">RELIEF & GOVERNANCE</div>
  <a href="shelters.html"><span class="nav-icon">⌂</span>Shelters</a>
  <a href="family-reunification.html"><span class="nav-icon">♡</span>Family Reunification</a>
  <a href="human-approval.html"><span class="nav-icon">✓</span>Human Approval</a>
  <a href="audit-logs.html"><span class="nav-icon">◷</span>Audit Logs</a>
  <a href="reports.html"><span class="nav-icon">▤</span>Reports</a>
 </nav>
 <div class="side-footer">
  <div class="avatar">ME</div>
  <div><strong>Medical Intelligence</strong><small>Synthetic Demo Workspace</small></div>
 </div>
</aside>

<main class="main">
<header class="topbar">
 <div class="top-left">
  <button class="btn menu-btn" id="menuBtn" aria-label="Toggle menu">☰</button>
  <div class="page-icon">✚</div>
  <div><h1>Medical Emergency</h1><p>Case triage · Medical coordination · Human review</p></div>
 </div>
 <div class="top-actions">
  <span class="badge"><span class="dot"></span>DEMO MODE</span>
  <button class="btn" id="resetBtn">↻ Reset</button>
  <button class="btn gold" id="exportBtn">↓ Export CSV</button>
 </div>
</header>

<div class="content">
<div class="heading">
 <div>
  <div class="eyebrow">EMERGENCY HEALTHCARE COORDINATION</div>
  <h2>Medical Emergency Intelligence</h2>
  <p>Register fictional medical cases, explore resource scenarios and review proposed actions.</p>
 </div>
 <span class="tag">HUMAN-SUPERVISED WORKFLOW</span>
</div>

<section class="hero">
 <div class="eyebrow">MEDICAL RESPONSE OPERATIONS</div>
 <h3>Coordinate Care. Prioritize Review. Maintain Accountability.</h3>
 <p>
  A proposed medical coordination layer for flood-related emergencies,
  integrating case intake, clinician-led triage, ambulance resource checks,
  hospital capacity verification, referral preparation and human approval.
  This standalone frontend contains no real medical decision engine.
 </p>
 <div class="hero-actions">
  <a class="btn gold" href="hospital-capacity.html">✚ Hospital Capacity</a>
  <a class="btn" href="ambulance-dispatch.html">▣ Ambulance Coordination</a>
  <a class="btn blue" href="human-approval.html">✓ Human Approval</a>
  <a class="btn red" href="tel:112">Emergency in India: 112</a>
 </div>
</section>

<section class="grid kpis">
 <div class="card kpi"><div class="kpi-head">Demo Medical Cases<div class="kpi-icon">▤</div></div><div class="kpi-number" id="kpiTotal">0</div><div class="kpi-caption"><b>SYNTHETIC</b> · Local case registry</div></div>
 <div class="card kpi"><div class="kpi-head">High / Critical Priority<div class="kpi-icon">⚑</div></div><div class="kpi-number" id="kpiUrgent">0</div><div class="kpi-caption"><b>SYNTHETIC</b> · User-assigned labels</div></div>
 <div class="card kpi"><div class="kpi-head">Pending Human Review<div class="kpi-icon">✓</div></div><div class="kpi-number" id="kpiPending">0</div><div class="kpi-caption"><b>DEMO ONLY</b> · No actual clinical review</div></div>
 <div class="card kpi"><div class="kpi-head">Sample Ambulances<div class="kpi-icon">▣</div></div><div class="kpi-number" id="kpiAmbulances">0</div><div class="kpi-caption"><b>FICTIONAL</b> · Not available for dispatch</div></div>
</section>

<section class="section grid columns">
 <div class="card panel">
  <div class="section-head">
   <div><div class="panel-title">Create Medical Coordination Scenario</div><div class="panel-sub">Use fictional case codes only. Do not enter real patient information.</div></div>
   <span class="tag">NEW CASE</span>
  </div>

  <form id="caseForm">
   <div class="form-grid">
    <div class="field">
     <label for="caseRef">Fictional Case Reference</label>
     <input class="input" id="caseRef" maxlength="40" placeholder="CASE-ALPHA" required>
    </div>
    <div class="field">
     <label for="district">District</label>
     <select class="select" id="district" required></select>
    </div>
    <div class="field">
     <label for="category">Medical Scenario Category</label>
     <select class="select" id="category" required>
      <option value="Injury">Injury</option>
      <option value="Water-Related Exposure">Water-Related Exposure</option>
      <option value="Chronic Care Support">Chronic Care Support</option>
      <option value="Mobility Assistance">Mobility Assistance</option>
      <option value="Medication Access">Medication Access</option>
      <option value="General Medical Assessment">General Medical Assessment</option>
     </select>
    </div>
    <div class="field">
     <label for="priority">Administrative Priority</label>
     <select class="select" id="priority">
      <option value="medium">Medium</option>
      <option value="low">Low</option>
      <option value="high">High</option>
      <option value="critical">Critical</option>
     </select>
    </div>
    <div class="field">
     <label for="care">Requested Coordination Type</label>
     <select class="select" id="care">
      <option>Medical Assessment Review</option>
      <option>Hospital Referral Review</option>
      <option>Transport Coordination Review</option>
      <option>Medication Support Review</option>
      <option>Accessibility Support Review</option>
     </select>
    </div>
    <div class="field">
     <label for="facility">Sample Facility</label>
     <select class="select" id="facility"></select>
    </div>
    <div class="field wide">
     <label for="notes">Scenario Notes</label>
     <textarea class="textarea" id="notes" maxlength="500" placeholder="Fictional scenario details only..."></textarea>
    </div>
   </div>

   <label class="checkline">
    <input type="checkbox" id="syntheticConfirm" required>
    <span>I confirm this is fictional training data and contains no real patient details or emergency request.</span>
   </label>
   <button class="btn gold" type="submit">+ Create Demo Case</button>
  </form>
 </div>

 <div class="card panel">
  <div class="panel-title">Medical Response Workflow</div>
  <div class="panel-sub" style="margin-bottom:17px">Proposed production sequence. No operational actions are performed.</div>
  <div class="info-list" id="workflowList"></div>

  <div class="notice">
   <strong>Clinical Safety Requirement</strong>
   <div>Priority labels in this demonstration are administrative examples, not validated medical triage scores. Diagnosis, triage, transport suitability and care decisions require qualified medical professionals.</div>
  </div>
 </div>
</section>

<section class="section">
 <div class="section-head">
  <div><h3>Medical Case Registry</h3><p>Search, inspect and simulate administrative review of fictional cases.</p></div>
  <span class="tag" id="caseCount">0 CASES</span>
 </div>
 <div class="card panel">
  <div class="filters">
   <input class="input" id="searchInput" placeholder="Search case reference, district or category..." aria-label="Search cases">
   <select class="select" id="priorityFilter">
    <option value="all">All Priorities</option>
    <option value="critical">Critical</option>
    <option value="high">High</option>
    <option value="medium">Medium</option>
    <option value="low">Low</option>
   </select>
   <select class="select" id="statusFilter">
    <option value="all">All Statuses</option>
    <option value="pending">Pending</option>
    <option value="reviewed">Reviewed</option>
    <option value="approved">Demo Approved</option>
    <option value="rejected">Demo Rejected</option>
    <option value="closed">Closed</option>
   </select>
  </div>
  <div class="table-wrap">
   <table>
    <thead><tr><th>Case ID</th><th>District</th><th>Category</th><th>Priority</th><th>Status</th><th>Facility</th><th>Action</th></tr></thead>
    <tbody id="caseTable"></tbody>
   </table>
  </div>
 </div>
</section>

<section class="section grid two-col">
 <div class="card panel">
  <div class="section-head">
   <div><div class="panel-title">Fictional Hospital Resources</div><div class="panel-sub">Illustrative numbers, not verified availability or reservations.</div></div>
   <a class="btn" href="hospital-capacity.html">Full Dashboard →</a>
  </div>
  <div id="hospitalList"></div>
 </div>
 <div class="card panel">
  <div class="section-head">
   <div><div class="panel-title">Sample Ambulance Coordination</div><div class="panel-sub">Fictional vehicles and statuses. No actual dispatch capability.</div></div>
   <a class="btn" href="ambulance-dispatch.html">Dispatch Concept →</a>
  </div>
  <div id="ambulanceList"></div>
 </div>
</section>

<section class="section grid columns">
 <div class="card panel">
  <div class="panel-title">Medical Agent Architecture</div>
  <div class="panel-sub" style="margin-bottom:16px">Planned backend integrations and human-controlled workflow.</div>
  <div class="info-list" id="agentList"></div>
 </div>
 <div class="card panel">
  <div class="panel-title">Activity Timeline</div>
  <div class="panel-sub" style="margin-bottom:19px">Events generated inside this browser demonstration.</div>
  <div class="timeline" id="timeline"></div>
 </div>
</section>

<div class="notice">
 <strong>⚠ Medical Emergency Disclaimer.</strong>
 <div>
  This page is a software portfolio demonstration and must not be used to assess real patients, arrange transport or locate available care. No real alerts, ambulance requests, medical referrals or hospital bookings are sent. In an actual emergency in India, contact <a href="tel:112" style="color:var(--gold2);font-weight:800">112</a> and follow professional emergency guidance.
 </div>
</div>
<footer class="footer">KERALA FLOOD RELIEF COORDINATION AGENT · MEDICAL EMERGENCY DEMONSTRATION · SYNTHETIC DATA</footer>
</div>
</main>
</div>

<div class="overlay" id="modalOverlay" role="presentation">
 <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
  <div class="modal-head">
   <h3 id="modalTitle">Medical Case Review</h3>
   <button class="btn" id="closeModal" aria-label="Close dialog">✕</button>
  </div>
  <p class="panel-sub">All actions below update local demonstration state only.</p>
  <div class="detail-grid" id="modalDetails"></div>
  <div class="notice">
   <strong>Human Oversight</strong>
   <div>Demo approval does not authorize treatment, hospital transfer, referral, transport or dispatch.</div>
  </div>
  <div class="modal-actions" id="modalActions"></div>
 </div>
</div>

<script>
"use strict";

const $=id=>document.getElementById(id);

const DISTRICTS=[
 "Thiruvananthapuram","Kollam","Pathanamthitta","Alappuzha",
 "Kottayam","Idukki","Ernakulam","Thrissur","Palakkad",
 "Malappuram","Kozhikode","Wayanad","Kannur","Kasaragod"
];

const HOSPITALS=[
 {id:"HSP-001",name:"Demo Medical Center Alpha",district:"Thiruvananthapuram",beds:44,total:180,icu:7,icuTotal:24},
 {id:"HSP-002",name:"Demo Medical Center Beta",district:"Ernakulam",beds:30,total:210,icu:4,icuTotal:30},
 {id:"HSP-003",name:"Demo Medical Center Gamma",district:"Thrissur",beds:48,total:145,icu:8,icuTotal:20},
 {id:"HSP-004",name:"Demo Medical Center Delta",district:"Wayanad",beds:4,total:90,icu:0,icuTotal:12},
 {id:"HSP-005",name:"Demo Medical Center Epsilon",district:"Kozhikode",beds:30,total:170,icu:4,icuTotal:22}
];

const AMBULANCES=[
 {id:"AMB-DEMO-01",district:"Wayanad",status:"Sample Available",type:"Basic Support"},
 {id:"AMB-DEMO-02",district:"Thrissur",status:"Sample Available",type:"Advanced Support"},
 {id:"AMB-DEMO-03",district:"Ernakulam",status:"Fictional Assigned",type:"Basic Support"},
 {id:"AMB-DEMO-04",district:"Alappuzha",status:"Sample Available",type:"Basic Support"},
 {id:"AMB-DEMO-05",district:"Kozhikode",status:"Fictional Maintenance",type:"Advanced Support"}
];

const SEED_CASES=[
 {id:"MED-DEMO-001",ref:"CASE-ALPHA",district:"Wayanad",category:"Injury",priority:"critical",care:"Medical Assessment Review",facility:"HSP-004",notes:"Fictional incident scenario.",status:"pending"},
 {id:"MED-DEMO-002",ref:"CASE-BETA",district:"Thrissur",category:"Medication Access",priority:"medium",care:"Medication Support Review",facility:"HSP-003",notes:"Fictional medication logistics scenario.",status:"reviewed"},
 {id:"MED-DEMO-003",ref:"CASE-GAMMA",district:"Ernakulam",category:"Water-Related Exposure",priority:"high",care:"Hospital Referral Review",facility:"HSP-002",notes:"Fictional coordination scenario.",status:"pending"},
 {id:"MED-DEMO-004",ref:"CASE-DELTA",district:"Alappuzha",category:"Mobility Assistance",priority:"low",care:"Accessibility Support Review",facility:"HSP-001",notes:"Fictional support scenario.",status:"approved"}
];

const WORKFLOW=[
 ["01 · Case Intake","Register fictional incident references and scenario details."],
 ["02 · Identity & Source Verification","Verify authenticity and consent in a production system."],
 ["03 · Clinician-Led Triage","Qualified personnel determine real clinical urgency."],
 ["04 · Resource Verification","Confirm actual facility and transport availability."],
 ["05 · Human Approval","Authorize consequential actions using verified information."],
 ["06 · Dispatch & Follow-Up","Production-only execution, acknowledgment and audit."],
];

const AGENTS=[
 ["Medical Response Agent","Proposed clinical coordination assistant; no autonomous diagnosis."],
 ["SOS Triage Agent","Proposed incident categorization with uncertainty and escalation."],
 ["Shelter Allocation Agent","Proposed accessibility and humanitarian coordination."],
 ["Route Planning Agent","Proposed verified route and transport decision support."],
 ["Verification Agent","Proposed source checks, duplicate detection and validation."],
 ["Human Approval Agent","Proposed mandatory approval and auditable decisions."]
];

const state={
 cases:structuredClone(SEED_CASES),
 nextId:5,
 selectedId:null,
 events:[
  {title:"Medical demo initialized",detail:"Synthetic medical coordination dataset loaded.",time:"Demo"},
  {title:"Human review required",detail:"No medical action can be executed from this page.",time:"Demo"}
 ]
};

function element(tag,className,text){
 const node=document.createElement(tag);
 if(className)node.className=className;
 if(text!==undefined)node.textContent=String(text);
 return node;
}

function toast(message){
 document.querySelector(".toast")?.remove();
 const node=element("div","toast",message);
 document.body.appendChild(node);
 setTimeout(()=>node.remove(),3000);
}

function logEvent(title,detail){
 state.events.unshift({
  title,detail,time:new Date().toLocaleTimeString("en-IN")
 });
 state.events=state.events.slice(0,9);
 renderTimeline();
}

function renderTimeline(){
 const container=$("timeline");
 container.replaceChildren();
 state.events.forEach(event=>{
  const row=element("div","event");
  const icon=element("div","event-icon","✚");
  const body=element("div");
  body.append(
   element("strong","",event.title),
   element("p","",event.detail),
   element("small","",event.time)
  );
  row.append(icon,body);
  container.appendChild(row);
 });
}

function renderKPIs(){
 $("kpiTotal").textContent=state.cases.length;
 $("kpiUrgent").textContent=state.cases.filter(c=>
  ["critical","high"].includes(c.priority)&&c.status!=="closed"
 ).length;
 $("kpiPending").textContent=state.cases.filter(c=>c.status==="pending").length;
 $("kpiAmbulances").textContent=AMBULANCES.filter(a=>a.status==="Sample Available").length;
 $("caseCount").textContent=state.cases.length+" CASES";
}

function pill(value){
 return element("span","pill "+value,value.toUpperCase());
}

function renderCases(){
 const query=$("searchInput").value.trim().toLowerCase();
 const priority=$("priorityFilter").value;
 const status=$("statusFilter").value;

 const filtered=state.cases.filter(c=>
  [c.id,c.ref,c.district,c.category].join(" ").toLowerCase().includes(query)&&
  (priority==="all"||c.priority===priority)&&
  (status==="all"||c.status===status)
 );

 const tbody=$("caseTable");
 tbody.replaceChildren();

 if(!filtered.length){
  const row=element("tr");
  const cell=element("td","","No matching fictional cases.");
  cell.colSpan=7;
  row.appendChild(cell);
  tbody.appendChild(row);
  return;
 }

 filtered.forEach(c=>{
  const row=element("tr");
  const fields=[c.id+" · "+c.ref,c.district,c.category];

  fields.forEach(value=>row.appendChild(element("td","",value)));

  const priorityCell=element("td");
  priorityCell.appendChild(pill(c.priority));
  row.appendChild(priorityCell);

  const statusCell=element("td");
  statusCell.appendChild(pill(c.status));
  row.appendChild(statusCell);

  const facility=HOSPITALS.find(h=>h.id===c.facility);
  row.appendChild(element("td","",facility?.name||"Unassigned"));

  const action=element("td");
  const button=element("button","small-btn","Inspect →");
  button.addEventListener("click",()=>openModal(c.id));
  action.appendChild(button);
  row.appendChild(action);
  tbody.appendChild(row);
 });
}

function renderHospitals(){
 const container=$("hospitalList");
 container.replaceChildren();

 HOSPITALS.forEach(h=>{
  const card=element("div","resource");
  const head=element("div","resource-head");
  head.append(
   element("strong","",h.name),
   element("span","tag",h.district)
  );

  const detail=element(
   "p","",
   `Synthetic general beds: ${h.beds}/${h.total} · ICU: ${h.icu}/${h.icuTotal}`
  );

  const progress=element("div","progress");
  const bar=element("span");
  bar.style.width=(100*(h.total-h.beds)/h.total)+"%";
  progress.appendChild(bar);

  card.append(head,detail,progress);
  container.appendChild(card);
 });
}

function renderAmbulances(){
 const container=$("ambulanceList");
 container.replaceChildren();

 AMBULANCES.forEach(a=>{
  const row=element("div","info-row");
  const body=element("div");
  body.append(
   element("strong","",a.id),
   element("small","",a.district+" · "+a.type)
  );

  const status=element("span","pill "+(
   a.status==="Sample Available"?"approved":
   a.status==="Fictional Assigned"?"pending":"rejected"
  ),a.status);

  row.append(body,status);
  container.appendChild(row);
 });
}

function renderStaticLists(){
 const workflow=$("workflowList");
 WORKFLOW.forEach(([title,description])=>{
  const row=element("div","info-row");
  const body=element("div");
  body.append(element("strong","",title),element("small","",description));
  row.appendChild(body);
  workflow.appendChild(row);
 });

 const agents=$("agentList");
 AGENTS.forEach(([title,description])=>{
  const row=element("div","info-row");
  const body=element("div");
  body.append(element("strong","",title),element("small","",description));
  row.appendChild(body);
  agents.appendChild(row);
 });
}

function populateSelects(){
 const district=$("district");
 district.appendChild(new Option("Select District","",true,true));
 DISTRICTS.forEach(name=>district.appendChild(new Option(name,name)));
 district.value="";

 const facility=$("facility");
 facility.appendChild(new Option("No facility selected",""));
 HOSPITALS.forEach(h=>{
  facility.appendChild(new Option(h.name+" · "+h.district,h.id));
 });
}

function createCase(event){
 event.preventDefault();

 const ref=$("caseRef").value.trim();
 const district=$("district").value;
 const category=$("category").value;
 const priority=$("priority").value;
 const care=$("care").value;
 const facility=$("facility").value;
 const notes=$("notes").value.trim();

 if(!ref||!district){
  toast("Enter a fictional case reference and district.");
  return;
 }

 if(!$("syntheticConfirm").checked){
  toast("Confirm that all information is fictional.");
  return;
 }

 const id="MED-DEMO-"+String(state.nextId++).padStart(3,"0");

 state.cases.unshift({
  id,ref,district,category,priority,care,facility,notes,status:"pending"
 });

 $("caseForm").reset();
 renderKPIs();
 renderCases();
 logEvent("Fictional case created",id+" added for administrative review.");
 toast("Demo medical case created");
}

function addDetail(container,label,value){
 const card=element("div","detail");
 card.append(
  element("small","",label),
  element("strong","",value||"Not specified")
 );
 container.appendChild(card);
}

function openModal(id){
 const c=state.cases.find(item=>item.id===id);
 if(!c)return;

 state.selectedId=id;

 const details=$("modalDetails");
 details.replaceChildren();

 const facility=HOSPITALS.find(h=>h.id===c.facility);

 [
  ["Case ID",c.id],
  ["Fictional Reference",c.ref],
  ["District",c.district],
  ["Category",c.category],
  ["Priority",c.priority.toUpperCase()],
  ["Status",c.status.toUpperCase()],
  ["Coordination Request",c.care],
  ["Sample Facility",facility?.name||"Not selected"],
  ["Scenario Notes",c.notes||"None"]
 ].forEach(([label,value])=>addDetail(details,label,value));

 const actions=$("modalActions");
 actions.replaceChildren();

 const close=element("button","btn","Close");
 close.addEventListener("click",closeModal);
 actions.appendChild(close);

 if(c.status==="pending"){
  addAction(actions,"Mark Demo Reviewed","reviewed","btn blue");
 }
 if(c.status==="reviewed"){
  addAction(actions,"Simulate Approval","approved","btn gold");
  addAction(actions,"Simulate Rejection","rejected","btn red");
 }
 if(c.status!=="closed"){
  addAction(actions,"Close Demo Case","closed","btn");
 }

 $("modalOverlay").classList.add("open");
 $("closeModal").focus();
}

function addAction(container,label,newStatus,className){
 const button=element("button",className,label);
 button.addEventListener("click",()=>changeStatus(newStatus));
 container.appendChild(button);
}

function changeStatus(newStatus){
 const c=state.cases.find(item=>item.id===state.selectedId);
 if(!c)return;

 const valid={
  pending:["reviewed","closed"],
  reviewed:["approved","rejected","closed"],
  approved:["closed"],
  rejected:["closed"],
  closed:[]
 };

 if(!valid[c.status].includes(newStatus)){
  toast("This demo status transition is not allowed.");
  return;
 }

 c.status=newStatus;
 renderKPIs();
 renderCases();
 logEvent("Demo case status updated",`${c.id} → ${newStatus}. No real action taken.`);
 closeModal();
 toast("Local demo status updated");
}

function closeModal(){
 $("modalOverlay").classList.remove("open");
 state.selectedId=null;
}

function csvEscape(value){
 let text=String(value??"");
 if(/^\s*[=+\-@]/.test(text))text="'"+text;
 return '"'+text.replace(/"/g,'""')+'"';
}

function exportCSV(){
 const fields=["id","ref","district","category","priority","care","facility","status","notes"];
 const rows=[
  fields.map(csvEscape).join(","),
  ...state.cases.map(c=>fields.map(field=>csvEscape(c[field])).join(","))
 ];

 const blob=new Blob(["\uFEFF"+rows.join("\r\n")],{
  type:"text/csv;charset=utf-8"
 });

 const url=URL.createObjectURL(blob);
 const link=document.createElement("a");
 link.href=url;
 link.download="medical-emergency-synthetic-cases.csv";
 document.body.appendChild(link);
 link.click();
 link.remove();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
 toast("Synthetic cases exported");
}

function resetDemo(){
 state.cases=structuredClone(SEED_CASES);
 state.nextId=5;
 state.events=[{
  title:"Medical demo reset",
  detail:"Original synthetic medical cases restored.",
  time:new Date().toLocaleTimeString("en-IN")
 }];
 $("caseForm").reset();
 $("searchInput").value="";
 $("priorityFilter").value="all";
 $("statusFilter").value="all";
 closeModal();
 renderKPIs();
 renderCases();
 renderTimeline();
 toast("Medical demonstration reset");
}

function initialize(){
 populateSelects();
 renderStaticLists();
 renderHospitals();
 renderAmbulances();
 renderKPIs();
 renderCases();
 renderTimeline();

 $("caseForm").addEventListener("submit",createCase);
 $("searchInput").addEventListener("input",renderCases);
 $("priorityFilter").addEventListener("change",renderCases);
 $("statusFilter").addEventListener("change",renderCases);

 $("exportBtn").addEventListener("click",exportCSV);
 $("resetBtn").addEventListener("click",resetDemo);
 $("closeModal").addEventListener("click",closeModal);

 $("modalOverlay").addEventListener("click",event=>{
  if(event.target===$("modalOverlay"))closeModal();
 });

 document.addEventListener("keydown",event=>{
  if(event.key==="Escape")closeModal();
 });

 $("menuBtn").addEventListener("click",()=>{
  $("sidebar").classList.toggle("open");
 });
}

initialize();
</script>
</body>
</html>
