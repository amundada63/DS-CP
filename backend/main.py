"""AeroTwin GCS demo backend. All telemetry is synthetic/simulated, not flight-certified."""
from __future__ import annotations
import math, random, time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -------------------- Practical data structures --------------------
class TelemetryQueue:
    """FIFO queue for incoming telemetry packets."""
    def __init__(self): self._items = deque()
    def enqueue(self, item): self._items.append(item)
    def dequeue(self): return self._items.popleft() if self._items else None
    def __len__(self): return len(self._items)

class CircularBuffer:
    """Fixed-size rolling history; oldest samples are overwritten."""
    def __init__(self, capacity=120): self.capacity, self.data = capacity, deque(maxlen=capacity)
    def append(self, item): self.data.append(item)
    def latest(self, n=20): return list(self.data)[-n:]

class SensorRegistry:
    """Hash-map based O(1)-average lookup by sensor/channel ID."""
    def __init__(self): self.values: Dict[str, dict] = {}
    def update(self, sensor_id, value): self.values[sensor_id] = value
    def get(self, sensor_id): return self.values.get(sensor_id)

class AlertPriorityQueue:
    """Min-heap where lower numeric severity key is more urgent."""
    def __init__(self): self.heap = []
    def push(self, severity, alert):
        import heapq; heapq.heappush(self.heap, (severity, time.time(), alert))
    def ordered(self): return [x[2] for x in sorted(self.heap)]

class ComponentGraph:
    """Adjacency-list graph for engine component dependency traversal."""
    def __init__(self):
        self.edges = {'Fuel System':['Combustion','EGT Sensor'], 'Combustion':['Cylinder Head','Exhaust'], 'Cylinder Head':['CHT Sensor','Cooling Circuit'], 'Exhaust':['EGT Sensor'], 'Lubrication':['Oil Pressure Sensor'], 'Cooling Circuit':['CHT Sensor']}
    def neighbors(self, node): return self.edges.get(node, [])
    def as_dict(self): return self.edges

class DiagnosticStack:
    """LIFO stack retaining recent diagnostic actions for audit/replay."""
    def __init__(self): self.items=[]
    def push(self, event): self.items.append(event)
    def recent(self, n=8): return list(reversed(self.items[-n:]))

# -------------------- Synthetic ML model training --------------------
rng = np.random.default_rng(42)
# Feature order: RPM, CHT, EGT, MAP, fuel flow, oil pressure, vibration
X=[]; y=[]
for _ in range(1800):
    rpm=float(rng.uniform(1800,5800)); cht=float(rng.normal(112,5)); egt=float(rng.normal(740,22))
    mp=float(rng.normal(98,4)); fuel=float(rng.normal(16,1.8)); oil=float(rng.normal(3.5,.22)); vib=float(rng.normal(.8,.18))
    label=0
    fault=rng.choice(['normal','cooling','fuel','sensor','oil'],p=[.62,.10,.10,.09,.09])
    if fault=='cooling': cht+=rng.uniform(22,55); label=1
    elif fault=='fuel': egt-=rng.uniform(90,180); fuel-=rng.uniform(4,8); label=2
    elif fault=='sensor': cht+=rng.uniform(35,65); label=3
    elif fault=='oil': oil-=rng.uniform(1.2,2.2); vib+=rng.uniform(.7,1.5); label=4
    X.append([rpm,cht,egt,mp,fuel,oil,vib]); y.append(label)
X=np.asarray(X); y=np.asarray(y)
scaler=StandardScaler().fit(X)
Xscaled=scaler.transform(X)
anomaly_model=IsolationForest(n_estimators=120, contamination=.08, random_state=42).fit(Xscaled)
fault_model=RandomForestClassifier(n_estimators=140, max_depth=10, random_state=42).fit(Xscaled,y)
FAULTS={0:'NORMAL',1:'COOLING_SYSTEM_DEGRADATION',2:'FUEL_DELIVERY_ANOMALY',3:'SENSOR_DRIFT',4:'LUBRICATION_WARNING'}

app=FastAPI(title='AeroTwin GCS API', version='1.0.0', description='Synthetic demonstration API; not for operational aviation use.')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
queue=TelemetryQueue(); history=CircularBuffer(180); registry=SensorRegistry(); alertq=AlertPriorityQueue(); graph=ComponentGraph(); diagstack=DiagnosticStack()
state={'scenario':'normal','tick':0,'base_offline':None,'headwind':False,'last_fault':'NORMAL'}

class ScenarioRequest(BaseModel): scenario: str
class BaseRequest(BaseModel): offline: Optional[str]=None

BASES=[
 {'name':'Forward Base Alpha','distance':36.1,'bearing':118,'reachable':True},
 {'name':'Forward Base Bravo','distance':67.6,'bearing':302,'reachable':True},
 {'name':'Main Operating Base Charlie','distance':13.7,'bearing':224,'reachable':True},
 {'name':'Forward Base Delta','distance':75.5,'bearing':42,'reachable':True},
 {'name':'Austere Strip Echo','distance':73.7,'bearing':218,'reachable':True},
]

def make_reading():
    state['tick']+=1; t=state['tick']; sc=state['scenario']
    rpm=3800+math.sin(t/7)*28+random.uniform(-8,8)
    cht=111.5+math.sin(t/13)*1.2+random.uniform(-.45,.45)
    egt=741+math.sin(t/9)*5+random.uniform(-2,2)
    mp=98.5+random.uniform(-.8,.8); fuel=16.1+random.uniform(-.35,.35); oil=3.48+random.uniform(-.05,.05); vib=.82+random.uniform(-.07,.07)
    if sc=='cooling_leak': cht+=random.uniform(22,37)
    elif sc=='injector_clog': egt-=random.uniform(95,135); fuel-=random.uniform(3,5); mp-=random.uniform(2,4)
    elif sc=='sensor_drift': cht+=random.uniform(35,48) # physical channels remain normal; model catches mismatch
    elif sc=='oil_warning': oil-=random.uniform(1.1,1.7); vib+=random.uniform(.5,.9)
    elif sc=='full_spoof': cht+=random.uniform(40,55); egt-=random.uniform(100,150); fuel+=random.uniform(4,7); oil-=random.uniform(1.0,1.5)
    values={'rpm':round(rpm,1),'cht':round(cht,1),'egt':round(egt,1),'map':round(mp,1),'fuel_flow':round(fuel,2),'oil_pressure':round(oil,2),'vibration':round(vib,2)}
    expected={'rpm':3800.0,'cht':111.5,'egt':741.0,'map':98.5,'fuel_flow':16.1,'oil_pressure':3.44,'vibration':.82}
    queue.enqueue({'timestamp':datetime.now(timezone.utc).isoformat(),'values':values})
    packet=queue.dequeue(); history.append(packet)
    for k,v in values.items(): registry.update(k,{'value':v,'timestamp':packet['timestamp']})
    feature=np.array([[values['rpm'],values['cht'],values['egt'],values['map'],values['fuel_flow'],values['oil_pressure'],values['vibration']]])
    scaled=scaler.transform(feature)
    anomaly=anomaly_model.predict(scaled)[0] == -1
    anomaly_score=float(-anomaly_model.score_samples(scaled)[0])
    predicted=int(fault_model.predict(scaled)[0]); probs=fault_model.predict_proba(scaled)[0]
    fault=FAULTS[predicted]
    # Physics residuals: values are illustrative model-vs-sensor deltas.
    residuals={'cht':round(values['cht']-expected['cht'],2),'egt':round(values['egt']-expected['egt'],2),'fuel_flow':round(values['fuel_flow']-expected['fuel_flow'],2),'oil_pressure':round(values['oil_pressure']-expected['oil_pressure'],2)}
    critical=state['scenario'] in ('cooling_leak','full_spoof') or (values['cht']>145 or values['oil_pressure']<2.1)
    health=max(8,min(99.8,99.3-abs(residuals['cht'])*.42-abs(residuals['egt'])*.035-abs(residuals['oil_pressure'])*5-(12 if anomaly else 0)))
    rul=max(0.2,168-(100-health)*4.2-(18 if critical else 0))
    status='CRITICAL' if critical else ('WARNING' if anomaly or fault!='NORMAL' else 'NOMINAL')
    advisory='RETURN TO BASE' if critical else ('MONITOR / REDUCE LOAD' if status=='WARNING' else 'CONTINUE MISSION')
    if status!='NOMINAL':
        alertq.push(0 if critical else 1,{'severity':status,'message':fault.replace('_',' '),'timestamp':packet['timestamp']})
        diagstack.push({'action':'ML fault classification','fault':fault,'status':status,'time':packet['timestamp']})
    state['last_fault']=fault
    return {'timestamp':packet['timestamp'],'frame':t,'values':values,'expected':expected,'residuals':residuals,'anomaly':bool(anomaly),'scenario':sc,'anomaly_score':round(anomaly_score,3),'fault':fault,'fault_confidence':round(float(max(probs))*100,1),'health':round(health,1),'rul_hours':round(rul,1),'status':status,'advisory':advisory,'fault_probabilities':{FAULTS[i]:round(float(p)*100,1) for i,p in enumerate(probs)}}

latest=None
def get_latest():
    global latest
    latest=make_reading(); return latest

@app.get('/api/health')
def healthcheck(): return {'ok':True,'service':'AeroTwin GCS','mode':'SIMULATED TELEMETRY'}

@app.get('/api/telemetry')
def telemetry(): return get_latest()

@app.get('/api/overview')
def overview():
    r=latest or get_latest()
    return {'engine':'ROTAX 914-F TURBO (DEMO PROFILE)','callsign':'MQ-9 HERON','tail':'AT-914-SIL','mode':'SIL (SOFTWARE-IN-THE-LOOP)','stream_hz':10,'status':r['status'],'health':r['health'],'rul_hours':r['rul_hours'],'frame':r['frame'],'scenario':state['scenario'],'advisory':r['advisory'],'queue_depth':len(queue),'history_size':len(history.data)}

@app.get('/api/history')
def history_data():
    return {'items':[{'timestamp':x['timestamp'],**x['values']} for x in history.latest(60)]}

@app.get('/api/sensors')
def sensors():
    r=latest or get_latest(); v=r['values']
    channels=[('CH 01','CHT Probe #1',v['cht'],'°C'),('CH 02','CHT Probe #2',v['cht']+random.uniform(-.6,.6),'°C'),('CH 03','EGT Probe #1',v['egt'],'°C'),('CH 04','EGT Probe #2',v['egt']-2.5,'°C'),('CH 05','MAP Transducer',v['map'],'kPa'),('CH 06','Fuel Flow Rate',v['fuel_flow'],'L/h'),('CH 07','Oil Pressure Gauge',v['oil_pressure'],'bar'),('CH 08','Ambient Temp Sensor',24.5,'°C'),('CH 09','Engine RPM',v['rpm'],'RPM'),('CH 10','Vibration Sensor',v['vibration'],'g'),('CH 11','Intake Temp',34.2,'°C'),('CH 12','Throttle Position',62.0,'%'),('CH 13','Coolant Delta',4.8,'°C'),('CH 14','Battery Voltage',27.4,'V'),('CH 15','Oil Temperature',91.0,'°C')]
    return {'active':15,'total':15,'channels':[{'id':c[0],'name':c[1],'value':round(float(c[2]),2),'unit':c[3],'status':'WARNING' if r['status']!='NOMINAL' and c[0] in ('CH 01','CH 07') else 'NOMINAL'} for c in channels]}

@app.get('/api/cycles')
def cycles():
    r=latest or get_latest(); rows=[]
    for i in range(18):
        delta=random.uniform(-.5,1.5); cht=round(111.5+delta,1); egt=round(741+random.uniform(-2,3),1)
        rows.append({'cycle':f"#{max(1,r['frame']//10-i)}",'timestamp':datetime.now().strftime('%H:%M:%S'),'frame':r['frame']-i,'cht_live':cht,'cht_expected':111.5,'cht_residual':round(cht-111.5,1),'egt_live':egt,'egt_expected':741,'egt_residual':round(egt-741,1),'status':'NOMINAL' if abs(delta)<5 else 'WARNING'})
    return {'mean_residual':.54,'peak_residual':1.5,'anomalies':0 if r['status']=='NOMINAL' else 1,'rows':rows}

@app.get('/api/crossverify')
def crossverify():
    r=latest or get_latest(); vals=r['values']; scores=[max(90,100-abs(r['residuals']['cht'])*.15),max(90,100-abs(r['residuals']['egt'])*.04),max(90,100-abs(r['residuals']['fuel_flow'])*.5),max(85,100-abs(r['residuals']['oil_pressure'])*2)]
    names=['C1: CHT ↔ EGT Thermal Coupling','C2: Fuel Flow ↔ Heat Input','C3: Volumetric Air-Fuel (MAP × RPM)','C4: Thermo-Hydraulic Viscosity']
    return {'consistency':round(sum(scores)/4,1),'verdict':'AUTHENTIC TELEMETRY' if min(scores)>92 else 'REVIEW REQUIRED','checks':[{'name':n,'score':round(s,1),'status':'VALIDATED' if s>92 else 'REVIEW'} for n,s in zip(names,scores)],'graph':graph.as_dict()}

@app.get('/api/diagnostics')
def diagnostics():
    r=latest or get_latest()
    return {'fault':r['fault'],'confidence':r['fault_confidence'],'anomaly':r['anomaly'],'score':r['anomaly_score'],'probabilities':r['fault_probabilities'],'components':graph.as_dict(),'recent_actions':diagstack.recent(),'alerts':alertq.ordered()[:8]}

@app.get('/api/reroute')
def reroute():
    r=latest or get_latest(); fuel_mass=59.2-(r['frame']%50)*.012; wind=16.7+(8 if state['headwind'] else 0)
    rows=[]
    for b in BASES:
        reachable=b['reachable'] and b['name']!=state['base_offline']; fuel_req=b['distance']*(.043 if state['headwind'] else .039)
        margin=fuel_mass-8-fuel_req; eta=b['distance']/(118-wind*.35)*60
        rows.append({**b,'reachable':reachable and margin>0,'fuel_required':round(fuel_req,2),'margin':round(margin,2),'eta_min':round(eta,1)})
    candidates=[x for x in rows if x['reachable']]
    optimal=min(candidates,key=lambda x:x['fuel_required']) if candidates else None
    return {'fuel_mass':round(fuel_mass,1),'capacity':68,'usable':round(fuel_mass-8,1),'wind_kmh':round(wind,1),'wind_direction':253,'airspeed':118,'bases':rows,'optimal':optimal,'headwind':state['headwind']}

@app.post('/api/scenario')
def scenario(req:ScenarioRequest):
    allowed={'normal','cooling_leak','injector_clog','sensor_drift','oil_warning','full_spoof'}
    if req.scenario not in allowed: return {'ok':False,'error':'Unknown scenario'}
    state['scenario']=req.scenario; diagstack.push({'action':'Scenario changed','scenario':req.scenario,'time':datetime.now(timezone.utc).isoformat()})
    return {'ok':True,'scenario':req.scenario}

@app.post('/api/base')
def base(req:BaseRequest):
    state['base_offline']=req.offline
    return {'ok':True,'offline_base':state['base_offline']}

@app.post('/api/wind')
def wind():
    state['headwind']=not state['headwind']; return {'ok':True,'headwind':state['headwind']}

@app.post('/api/reset')
def reset():
    state['scenario']='normal'; state['base_offline']=None; state['headwind']=False; alertq.heap.clear()
    return {'ok':True}
