// /judge/ page: per-round Live view (original video + overlays) and Background flow.
const NS='http://www.w3.org/2000/svg';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fx=()=>window.BXFX||{count:(el,v,d)=>{el.textContent=Number(v).toFixed(d??2);},glitch(){},reveal(){},flowDots(){},on:()=>false};
const STAGE_X={detect:10,track:132,geometry:254,identity:376,finish:498};
const REASON_TEXT={
 two_bey_tracking_requires_review:'兩顆追蹤需覆核',two_bey_unassigned_identity_unknown:'有未指派觀測、身分不明',
 two_bey_singleton_segment:'只追到單顆片段',two_bey_unassigned_observations:'有未指派的觀測',
 calibration_low_score:'場地校正分數低',calibration_geometry_missing:'場地幾何缺失',calibration_relocated:'場地校正重新定位',
 identity_unanchored:'X／B 身分未錨定',track_missing:'軌跡缺失',track_evidence_missing:'軌跡證據缺失',
 track_lost:'追蹤中斷',track_evidence_gap:'軌跡證據有缺口',finish_unknown:'終結方式未知',id_swap_risk:'可能身分互換',
 stadium_motion_unknown:'場地晃動原因未知',stop_evidence_missing:'停轉證據缺失',
 landing_timing_unknown:'落地時間不明',launch_release_unobserved:'沒看到發射放開',shake_cause_unknown:'場地晃動原因不明',
 shake_by_bey_impact:'陀螺撞擊造成晃動',shake_not_by_launcher_body:'晃動非發射器造成',identity_after_merge_unknown:'交疊後身分不明',
 identity_recovered_after_merge:'交疊後身分已恢復（需確認）',air_evidence_incomplete:'空中證據不完整',pocket_entry_unconfirmed:'入袋未確認',
 exit_returned:'出界後又回場',position_unknown:'位置不明',track_evidence_duplicate:'軌跡證據重複',finish_candidate_voided:'終結候選被撤銷',
 after_finish:'終結後',X_stop_still:'停轉判定矛盾'
};
function reasonText(code){
 const [head,side]=String(code).split('@'),[base,...rest]=head.split(':');
 const t=REASON_TEXT[base]||(/_stop_/.test(base)?'停轉判定矛盾':null);
 if(!t)return null;
 const sub=base==='after_finish'?(reasonText(rest.join(':'))||rest.join(':')):'';
 const where=side?side.replace(/^([XB]):(\d+)$/,'$1 · f$2'):'';
 return `${t}${sub?`：${sub}`:''}${where?`（${where}）`:''}`;
}

let index=null,data=null,cache=new Map(),raf=0;

async function getJSON(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error(`${url} ${r.status}`);return r.json();}

function el(tag,attrs,parent){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs||{}))n.setAttribute(k,v);if(parent)parent.append(n);return n;}

function renderList(){
 const list=$('round-list');
 list.innerHTML=index.rounds.map(r=>`<button type="button" role="tab" class="${r.state}" data-key="${esc(r.key)}"><b>${esc(r.video.replace(/\.mp4$/,''))} · R${r.round}</b><small>${r.state==='review'?`覆核 · ${r.reason_count} 原因`:`${esc(r.finish_label)} · ${esc(r.winner||'—')}`}</small></button>`).join('');
 list.addEventListener('click',e=>{const b=e.target.closest('button[data-key]');if(b)location.hash=b.dataset.key;});
}

function flow(d){
 const svg=$('flow');svg.innerHTML='';
 const defs=el('defs',{},svg);
 const f=el('filter',{id:'glow',x:'-30%',y:'-30%',width:'160%',height:'160%'},defs);
 el('feGaussianBlur',{stdDeviation:'3',result:'b'},f);const m=el('feMerge',{},f);el('feMergeNode',{in:'b'},m);el('feMergeNode',{in:'SourceGraphic'},m);
 const stages=Object.fromEntries(d.stages.map(s=>[s.id,s]));
 const firstFlag=d.stages.find(s=>s.status==='flag');
 const edges=[];
 const keys=Object.keys(STAGE_X);
 keys.forEach((k,i)=>{if(i<keys.length-1)edges.push([`M${STAGE_X[k]+104} 52H${STAGE_X[keys[i+1]]}`,'']);});
 edges.push(['M602 52C620 52 618 28 636 28',d.state==='judged'?'':'dim']);
 edges.push(['M602 52C620 52 618 90 636 90',d.state==='review'?'':'dim']);
 edges.push(['M688 112V172H462','loop']);
 edges.push(['M300 172H62V76','loop']);
 for(const [p,c] of edges)el('path',{d:p,class:`edge ${c}`},svg);
 function node(x,y,w,h,label,sub,cls){const g=el('g',{class:`node ${cls}`},svg);el('rect',{x,y,width:w,height:h,rx:8},g);const t=el('text',{x:x+10,y:y+19},g);t.textContent=label;const s=el('text',{x:x+10,y:y+36,class:'sub'},g);s.textContent=sub;return g;}
 for(const k of keys){
  const s=stages[k]||{status:'ok',reasons:[],label:k};
  node(STAGE_X[k],30,104,44,s.label,s.status==='flag'?`! ${s.reasons.length} 項`:'✓ 通過',`${s.status} ${firstFlag&&firstFlag.id===k?'active':''}`);
 }
 node(636,8,116,40,'Output 判定',d.state==='judged'?'採用':'未採用',d.state==='judged'?'ok active':'idle');
 node(636,70,116,42,'Fallback 覆核',d.state==='review'?`${d.reasons.length} 原因`:'無',d.state==='review'?'flag active':'idle');
 node(300,150,162,44,'人工證據庫',`${Object.values(index.stores).reduce((a,b)=>a+b,0)} 份已存`,'store');
 if(stages.other)node(10,150,180,44,'其他原因',`! ${stages.other.reasons.length} 項`,'flag');
 const core=node(132,150,140,44,'核心引擎','locked · VLM 關','core');core.setAttribute('opacity','.85');
 fx().flowDots(svg);
}

function details(d){
 const conf=d.confidence;
 $('round-title').textContent=`${d.video} · 第 ${d.round} 回合`;
 const live=$('live');live.dataset.state=d.state;
 $('state-tag').textContent=d.state==='review'?'REVIEW · 待覆核':'EMIT · 已判定';
 $('detail-kv').innerHTML=[
  ['偵測器',d.detector.kind||'—'],['權重',d.detector.weights||'—'],
  ['場地校正',`${d.calibration.id||'—'} · ${d.calibration.kind||''} ${d.calibration.score!=null?Number(d.calibration.score).toFixed(2):''}`],
  ['幾何參照',d.geometry.status==='launch_reference'?`固定發射參照 f${d.geometry.sample_frame}；僅在該格顯示，非逐格對位`:'當格沒有可用幾何參照'],
  ['軌跡點',Object.entries(d.tracks).map(([k,v])=>`${k||'未指派'} ${v.length}`).join(' · ')||'無'],
  ['候選框影格',d.boxes.length],
  ['發射／終結',`${d.t_launch!=null?d.t_launch.toFixed(2):'—'}s → ${d.t_end!=null?d.t_end.toFixed(2):'—'}s`]
 ].map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
 const checks=$('checks');
 checks.innerHTML=d.stages.map(s=>`<li class="${s.status==='flag'?'flag':''}">${esc(s.label)}${s.status==='flag'?` <small class="c-orange">${s.reasons.length} 項</small>`:''}</li>`).join('');
 fx().reveal(checks);
 $('out-main').textContent=d.finish?`${d.finish_label} · ${d.winner||'—'}`:'未判定';
 $('out-sub').textContent=d.finish?`${d.points} 分${d.reviewed?' · 已人工覆核':''}`:'沒有可採用的自動勝負';
 const meter=$('meter');
 meter.querySelector('b').style.left=`${Math.min(100,d.threshold*100)}%`;
 meter.querySelector('i').style.width='0%';
 requestAnimationFrame(()=>requestAnimationFrame(()=>{meter.querySelector('i').style.width=`${Math.max(2,Math.min(100,conf*100))}%`;}));
 fx().count($('conf'),conf,2);
 $('th').textContent=d.threshold.toFixed(2);
 $('th-sign').textContent=d.above_threshold?'≥':'<';
 $('fb-state').textContent=d.state==='review'?'需要人工覆核':'未觸發';
 $('reasons').innerHTML=d.reasons.length?d.reasons.map(r=>`<li class="flag"><span>${esc(reasonText(r)||r)}${reasonText(r)?`<br><small class="mono c-grey">${esc(r)}</small>`:''}</span></li>`).join(''):'<li class="idle">沒有覆核原因</li>';
 fx().reveal($('reasons'));
 $('stores').innerHTML=Object.entries(index.stores).map(([k,v])=>`<li><span>${esc(k)}</span><b>${v}</b></li>`).join('')||'<li>無</li>';
 $('top-reasons').textContent=`整個 run：${index.review_count}／${index.round_count} 回合待覆核。`;
 const bar=$('bx-status');bar.dataset.state=d.state;
 const items=bar.querySelectorAll('span');
 const set=(i,label,val)=>{items[i].innerHTML=`${esc(label)} <b>${esc(val)}</b>`;};
 set(0,'回合',`${d.video.replace(/\.mp4$/,'')} R${d.round}`);
 set(1,'信心',`${conf.toFixed(2)} ${d.above_threshold?'≥':'<'} ${d.threshold.toFixed(2)}`);
 set(2,'Fallback',d.state==='review'?`ON · ${d.reasons.length}`:'OFF');
 set(3,'證據庫',Object.values(index.stores).reduce((a,b)=>a+b,0));
 if(!items[4]){const s=document.createElement('span');s.className='st-state';bar.append(s);}
 bar.querySelector('.st-state').textContent=d.state==='review'?'REVIEW':'EMIT';
 if(d.state==='review'){fx().glitch($('bx-title'));fx().glitch(bar.querySelector('.st-state'));}
 for(const b of document.querySelectorAll('#round-list button'))b.setAttribute('aria-current',String(b.dataset.key===d.key));
 const cur=document.querySelector(`#round-list button[data-key="${CSS.escape(d.key)}"]`),list=$('round-list');if(cur)list.scrollTo({left:cur.offsetLeft-list.offsetLeft-8,behavior:'auto'});
}

function setupOverlay(d){
 const svg=$('overlay');svg.innerHTML='';
 const [w,h]=d.coordinate_size;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('preserveAspectRatio','xMidYMid meet');
 const defs=el('defs',{},svg);
 const pat=el('pattern',{id:'hatch',width:'8',height:'8',patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'},defs);
 el('rect',{width:'8',height:'8',fill:'rgba(107,119,139,.10)'},pat);el('line',{x1:'0',y1:'0',x2:'0',y2:'8',stroke:'rgba(160,172,190,.45)','stroke-width':'2'},pat);
 const f=el('filter',{id:'glow',x:'-40%',y:'-40%',width:'180%',height:'180%'},defs);
 el('feGaussianBlur',{stdDeviation:'5',result:'b'},f);const m=el('feMerge',{},f);el('feMergeNode',{in:'b'},m);el('feMergeNode',{in:'SourceGraphic'},m);
 const zones=el('g',{id:'g-zones'},svg);
 for(const [name,pts] of Object.entries(d.geometry.zones||{}))el('polygon',{points:pts.map(p=>p.join(',')).join(' '),class:`ov-zone ${name}`},zones);
 el('g',{id:'g-trails'},svg);el('g',{id:'g-boxes'},svg);
}

function sourceBoxes(d,t){
 const b=d.boxes;if(!b.length)return null;
 const sourceFrame=Math.round(t*d.fps);
 let lo=0,hi=b.length-1;
 while(lo<hi){const mid=(lo+hi)>>1;if(b[mid][1]<sourceFrame)lo=mid+1;else hi=mid;}
 return b[lo][1]===sourceFrame?b[lo]:null;
}

function trailSegments(tracks,t){
 const segments=new Map();
 for(const [side,points] of Object.entries(tracks))for(const p of points){
  if(p[0]>t||p[0]<t-1.5)continue;
  if(!Number.isInteger(p[4])||!Number.isInteger(p[7]))throw Error('軌跡缺少實體／連續區間標記');
  const key=`${side}:${p[4]}:${p[7]}`;
  if(!segments.has(key))segments.set(key,{side,entity:p[4],points:[]});
  segments.get(key).points.push(p);
 }
 return Array.from(segments.values());
}

function draw(){
 if(!data)return;
 const d=data,t=$('video').currentTime-(d.media_offset||0);
 $('clock').textContent=`${t.toFixed(2)}s · f${Math.round(t*d.fps)}`;
 const trails=$('g-trails'),boxes=$('g-boxes');trails.innerHTML='';boxes.innerHTML='';
 $('g-zones').style.display=$('show-zones').checked&&d.geometry.status==='launch_reference'&&Math.round(t*d.fps)===d.geometry.sample_frame?'':'none';
 if($('show-trails').checked)for(const seg of trailSegments(d.tracks,t)){
  if(seg.points.length>1)el('polyline',{points:seg.points.map(p=>`${p[1]},${p[2]}`).join(' '),class:`ov-trail ${seg.side||'unknown'}`},trails);
 }
 const frame=sourceBoxes(d,t);
 if(frame){
  const items=frame[2].filter(b=>b[5]==='primary'||$('show-candidates').checked);
  const lead=items.filter(b=>b[5]==='primary').sort((a,b)=>b[4]-a[4])[0];
  for(const b of items.sort((a,b)=>(a[5]==='primary')-(b[5]==='primary'))){
   const [x1,y1,x2,y2,score,kind,label]=b,w=x2-x1,h=y2-y1,primary=kind==='primary';
   el('rect',{x:x1,y:y1,width:w,height:h,rx:3,class:`ov-box ${kind} ${b===lead?'lead':''}`},boxes);
   if(primary){
    el('ellipse',{cx:(x1+x2)/2,cy:(y1+y2)/2,rx:w*0.42,ry:h*0.42,class:'ov-ring'},boxes);
    if(b===lead){const k=Math.min(w,h)*0.3;el('path',{class:'ov-corner',d:[[x1-6,y1-6,1,1],[x2+6,y1-6,-1,1],[x1-6,y2+6,1,-1],[x2+6,y2+6,-1,-1]].map(([cx,cy,dx,dy])=>`M${cx+dx*k} ${cy}H${cx}V${cy+dy*k}`).join('')},boxes);}
   }
   const text=primary?`${label} · bey · ${score.toFixed(2)}`:`${String(label).replace(/_/g,' ')} ${score.toFixed(2)}`;
   const g=el('g',{class:`ov-label ${primary?'':'candidate'}`},boxes);
   // Keep labels readable when the frame is drawn small (narrow screens).
   const k=Math.max(1,Math.min(2.6,d.coordinate_size[0]/($('overlay').clientWidth||d.coordinate_size[0])*0.7));
   const fs=(primary?15:13)*k,th=fs+7*k,tw=text.length*fs*0.62+12*k,ly=Math.max(0,y1-th-4);
   const lx=Math.max(0,Math.min(x1,d.coordinate_size[0]-tw));
   el('rect',{x:lx,y:ly,width:tw,height:th,rx:3},g);
   const tx=el('text',{x:lx+6*k,y:ly+th-5*k,'font-size':fs},g);tx.style.fontSize=fs+'px';tx.textContent=text;
  }
 }
}

function loop(){draw();if(!$('video').paused)raf=requestAnimationFrame(loop);}

function setupVideo(d){
 const v=$('video'),empty=$('video-empty');
 cancelAnimationFrame(raf);
 const off=d.media_offset||0,start=Math.max(0,(d.t_launch??0)+off-1);
 empty.hidden=true;
 if(!d.src){v.removeAttribute('src');v.load();empty.hidden=false;empty.textContent='找不到原影片（不在本次 run 的 videos.json 允許清單）';draw();return;}
 if(v.getAttribute('src')!==d.src){v.src=d.src;}
 const seek=()=>{v.currentTime=start;draw();};
 if(v.readyState>=1)seek();else v.addEventListener('loadedmetadata',seek,{once:true});
 v.onerror=()=>{empty.hidden=false;empty.textContent='原影片無法載入；疊加資料仍可在暫停時檢視。';};
}

async function select(key){
 if(!index.rounds.some(r=>r.key===key))key=index.rounds[0]?.key;
 if(!key)return;
 if(!cache.has(key))cache.set(key,await getJSON(`data/${encodeURIComponent(key)}.json`));
 data=cache.get(key);
 details(data);flow(data);setupOverlay(data);setupVideo(data);
}

async function main(){
 try{index=await getJSON('index.json');}catch(e){$('round-title').textContent='無法讀取回合資料：'+e.message;return;}
 if(!index.rounds.length){$('round-title').textContent='這個 run 沒有回合';return;}
 renderList();
 const v=$('video');
 $('play').addEventListener('click',()=>{if(v.paused){v.play().catch(()=>{});}else v.pause();});
 v.addEventListener('play',()=>{$('play').textContent='暫停';loop();});
 v.addEventListener('pause',()=>{$('play').textContent='播放回合';draw();});
 v.addEventListener('seeked',draw);
 v.addEventListener('timeupdate',()=>{if(data&&data.t_end!=null&&v.currentTime-(data.media_offset||0)>data.t_end+1.5&&!v.paused)v.pause();});
 $('to-launch').addEventListener('click',()=>{if(data)v.currentTime=Math.max(0,(data.t_launch??0)+(data.media_offset||0)-0.2);});
 $('to-end').addEventListener('click',()=>{if(data)v.currentTime=Math.max(0,(data.t_end??0)+(data.media_offset||0)-0.2);});
 for(const id of ['show-candidates','show-zones','show-trails'])$(id).addEventListener('change',draw);
 document.addEventListener('bxfx',()=>{if(data)flow(data);});
 window.addEventListener('hashchange',()=>select(decodeURIComponent(location.hash.slice(1))));
 await select(decodeURIComponent(location.hash.slice(1)));
}
main();
