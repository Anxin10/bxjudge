export const options = {
 relation:[['unreviewed','尚未選擇'],['same','同一顆'],['different','不同顆'],['uncertain','看不清／無法判定']],
 scope:[['unreviewed','尚未選擇'],['same_round','同一回合'],['different_round','不同回合／重新發射'],['unknown','看不清／不知道']],
 side_before:[['unreviewed','未標記'],['X','X'],['B','B'],['unknown','不知道']],
 side_after:[['unreviewed','未標記'],['X','X'],['B','B'],['unknown','不知道']],
 visibility_before:[['unreviewed','未標記'],['visible','可見'],['partially_occluded','部分遮擋'],['unusable','畫面不可用'],['unknown','不知道']],
 visibility_after:[['unreviewed','未標記'],['visible','可見'],['partially_occluded','部分遮擋'],['unusable','畫面不可用'],['unknown','不知道']],
 collision:[['unreviewed','未標記'],['yes','確認有碰撞'],['no','確認沒有碰撞'],['unknown','不知道']]
};
export function blankCase(id){return {case_id:id,reviewed:false,...Object.fromEntries(Object.keys(options).map(k=>[k,'unreviewed'])),note:''};}
export function validateDraft(cases,manifest){
 if(!Array.isArray(cases)||cases.length!==manifest.cases.length)throw Error('草稿片段數量不符');
 for(let i=0;i<cases.length;i++){
  const c=cases[i];
  if(!c||c.case_id!==manifest.cases[i].case_id||typeof c.reviewed!=='boolean'||typeof c.note!=='string'||c.note.length>4000)throw Error('草稿欄位不符');
  if(Object.keys(c).sort().join('|')!==Object.keys(blankCase(c.case_id)).sort().join('|'))throw Error('草稿含未知欄位');
  for(const [k,values] of Object.entries(options))if(!values.some(v=>v[0]===c[k]))throw Error('草稿選項無效');
 }
 return cases;
}
export function stats(cases){return {reviewed:cases.filter(c=>c.reviewed).length,uncertain:cases.filter(c=>c.reviewed&&c.relation==='uncertain').length};}
export function payload(cases,sha,parent){
 for(const c of cases)if(c.reviewed&&(c.relation==='unreviewed'||c.scope==='unreviewed'))throw Error('請為已確認的組別選擇同物件關係及同回合狀態；不清楚可選未知');
 return {schema:'bxjudge.reid-review-labels.v1',manifest_sha256:sha,submitted_at:new Date().toISOString(),supersedes:parent,cases:structuredClone(cases)};
}
export function exportText(data){return JSON.stringify(data,null,2)+String.fromCharCode(10);}
const escapeText=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function mount(){
 const {manifest,sha}=JSON.parse(document.getElementById('data').textContent);
 const saveURL=new URL('save',document.baseURI).pathname,savedURL=new URL('saved',document.baseURI).pathname;
 const key='bxjudge.reid-review.'+sha;
 let cases=manifest.cases.map(c=>blankCase(c.case_id)),parent=null;
 let originalRaw=null,blocked=false,storageAvailable=true,persisted=false,saving=false;
 const status=document.getElementById('status'),error=document.getElementById('error');
 try{
  originalRaw=localStorage.getItem(key);
  if(originalRaw!==null){
   const draft=JSON.parse(originalRaw);
   if(!draft||draft.manifest_sha256!==sha||!(draft.parent===null||/^[a-f0-9]{64}$/.test(draft.parent)))throw Error('草稿版本或修訂來源不符');
   cases=validateDraft(draft.cases,manifest);parent=draft.parent;persisted=true;
  }
 }catch(e){
  if(originalRaw!==null){
   blocked=true;
   error.textContent='原草稿無法驗證，表單已鎖定，原始資料未覆寫：'+e.message+'。可匯出原始草稿，或明確載入專案已存版本。';
  }else{
   storageAvailable=false;
   error.textContent='此瀏覽器無法讀取持久草稿；變更只在記憶體，請勿直接關頁，請保存到專案或匯出備份。';
  }
 }
 const controls=document.getElementById('cases');
 function select(i,name,title){return `<label>${title}<select data-i="${i}" data-field="${name}">${options[name].map(([v,t])=>`<option value="${v}">${t}</option>`).join('')}</select></label>`;}
 function endpoint(c,which,label,kind){const e=c[which],b=e.box;const pad=45;const x=Math.max(0,b[0]-pad),y=Math.max(0,b[1]-pad),w=b[2]-b[0]+2*pad,h=b[3]-b[1]+2*pad;const bw=b[2]-b[0],bh=b[3]-b[1],k=Math.min(bw,bh)*0.28;const corners=[[b[0],b[1],1,1],[b[2],b[1],-1,1],[b[0],b[3],1,-1],[b[2],b[3],-1,-1]].map(([cx,cy,dx,dy])=>`M${cx+dx*k} ${cy}H${cx}V${cy+dy*k}`).join('');return `<figure class="ep ep-${kind}"><svg viewBox="${x} ${y} ${w} ${h}" role="img" aria-label="片段${label}來源影格${e.frame}"><image href="${escapeText(e.image)}" width="${c.source_size[0]}" height="${c.source_size[1]}" class="src"/><rect class="ov-box primary" x="${b[0]}" y="${b[1]}" width="${bw}" height="${bh}"/><path class="ov-corner" d="${corners}"/><g class="ov-label"><rect x="${b[0]}" y="${Math.max(y,b[1]-22)}" width="${Math.min(w,96)}" height="20"/><text x="${b[0]+5}" y="${Math.max(y,b[1]-22)+15}">${label} ${Number(e.score).toFixed(2)}</text></g></svg><figcaption>片段 ${label} · 來源 f${e.frame} · <a href="${escapeText(e.image)}">完整原尺寸畫面</a></figcaption></figure>`;}
 controls.innerHTML=manifest.cases.map((c,i)=>`<article class="panel bx-block case" data-i="${i}" data-state="review"><header><span class="block-label live">Case ${i+1}</span><h2>第 ${i+1} 組 · ${escapeText(c.video)}</h2><span class="tag state-text case-state">待核對</span></header><div class="case-grid"><div class="stage"><video controls preload="metadata" src="${escapeText(c.clip.path)}"></video><div class="scan" aria-hidden="true"></div></div><div class="endpoints">${endpoint(c,'before','①','a')}${endpoint(c,'after','②','b')}</div></div><p class="muted">比較青框①與珊瑚框②；①②是片段編號，和 X／B 側別無關。框是候選偵測，並非人工真值；影片左上角為原始來源影格。</p><div class="controls">${select(i,'relation','兩段是否同一顆？')}${select(i,'scope','前後是否屬同一回合？')}</div><details><summary>另標 X／B、可見性與碰撞（可以留空或選不知道）</summary><div class="controls">${select(i,'side_before','片段①身分')}${select(i,'side_after','片段②身分')}${select(i,'visibility_before','片段①可見性')}${select(i,'visibility_after','片段②可見性')}${select(i,'collision','這段是否確有碰撞？')}</div><label>補充說明<textarea data-i="${i}" data-field="note" maxlength="4000" rows="2"></textarea></label></details><label class="confirm"><input type="checkbox" data-i="${i}" data-field="reviewed">這組已核對（看不清也可以確認）</label><details><summary>來源與程式拒絕原因</summary><p>${escapeText(c.reasons.join('；'))}</p><small>原生 ID ${c.native_ids.join(' → ')}；候選 scope 尚未確認。你對同回合的標記不會自動建立全場範圍或改判決。</small></details></article>`).join('');
 function render(){
  for(const el of controls.querySelectorAll('[data-field]')){
   const c=cases[Number(el.dataset.i)];el.disabled=blocked;
   if(el.type==='checkbox')el.checked=c[el.dataset.field];else el.value=c[el.dataset.field];
  }
  document.getElementById('save').disabled=blocked||saving;
  document.getElementById('restore').disabled=saving||!saved.length;
  document.getElementById('export').textContent=blocked?'匯出原始草稿':'匯出 JSON 備份';
  const s=stats(cases);
  document.getElementById('progress').textContent=blocked?'原草稿待處理，核對表單鎖定':`已核對 ${s.reviewed}／${cases.length} 組；其中 ${s.uncertain} 組看不清（不是判對）`;
  for(const art of controls.querySelectorAll('article.case')){const c=cases[Number(art.dataset.i)];art.dataset.state=c.reviewed?'judged':'review';art.querySelector('.case-state').textContent=c.reviewed?'已核對':'待核對';}
  const bar=document.getElementById('reid-status');
  if(bar){bar.dataset.state=blocked||s.reviewed<cases.length?'review':'judged';const v=bar.querySelectorAll('b');if(v[0])v[0].textContent=blocked?'鎖定':`${s.reviewed}／${cases.length}`;}
 }
 function remember(){
  if(blocked)return false;
  if(!storageAvailable){persisted=false;return false;}
  try{
   localStorage.setItem(key,JSON.stringify({manifest_sha256:sha,parent,cases}));persisted=true;return true;
  }catch(e){
   persisted=false;storageAvailable=false;
   error.textContent='此瀏覽器無法持久保存草稿；目前變更只在記憶體，請保存到專案或匯出備份，請勿直接關頁。';
   return false;
  }
 }
 function markDirty(){
  if(blocked)return;
  status.textContent=remember()?'草稿已變更，尚未保存到專案。':'變更只在此頁記憶體，尚未保存到專案；請勿關頁，先保存或匯出備份。';
 }
 controls.addEventListener('change',e=>{
  const el=e.target;if(blocked||!el.dataset.field)return;
  cases[Number(el.dataset.i)][el.dataset.field]=el.type==='checkbox'?el.checked:el.value;
  markDirty();render();
 });
 controls.addEventListener('input',e=>{
  const el=e.target;if(blocked||el.dataset.field!=='note')return;
  cases[Number(el.dataset.i)].note=el.value;markDirty();
 });
 document.getElementById('save').addEventListener('click',async()=>{
  if(blocked||saving)return;
  error.textContent='';
  try{
   const p=payload(cases,sha,parent);if(!stats(cases).reviewed)throw Error('尚未確認任何一組');
   saving=true;render();
   const response=await fetch(saveURL,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
   const result=await response.json();
   if(!response.ok||!result.saved)throw Error(result.error||'沒有收到儲存成功回執');
   parent=result.sha256;remember();
   status.textContent=`已保存 ${result.reviewed_cases} 組到專案：${result.filename}；SHA-256 ${result.sha256}。舊版保留。`;
   if(JSON.stringify(cases)!==JSON.stringify(p.cases))status.textContent+=' 頁面另有較新的未存修改，請再次保存。';
   await refreshSaved().catch(()=>{error.textContent='已保存到專案，但版本清單更新失敗；請重新載入清單。';});
  }catch(e){
   error.textContent=e.message+(persisted?'；此瀏覽器版本仍在，尚未保存到專案。':'；目前變更只在記憶體，請勿關頁或重新載入，先匯出備份。');
  }finally{saving=false;render();}
 });
 document.getElementById('export').addEventListener('click',()=>{
  try{
   // Backup must also be possible for an incomplete or invalid draft.
   const data=blocked?originalRaw:exportText({schema:'bxjudge.reid-review-labels.v1',manifest_sha256:sha,submitted_at:new Date().toISOString(),supersedes:parent,cases:structuredClone(cases)});
   const url=URL.createObjectURL(new Blob([data],{type:'application/json'})),link=document.createElement('a');
   link.href=url;link.download=(blocked?'reid-original-draft-':'reid-labels-')+sha.slice(0,16)+'.json';link.click();
   setTimeout(()=>URL.revokeObjectURL(url),1000);
   status.textContent=blocked?'已要求下載原始草稿，原資料與表單鎖定均保留；這不代表專案保存。':'已要求下載JSON備份；這不代表已儲存到專案。';
  }catch(e){error.textContent=e.message;}
 });
 let saved=[];
 async function refreshSaved(){
  const response=await fetch(savedURL);if(!response.ok)throw Error('無法讀取已存版本清單');
  saved=(await response.json()).snapshots;const select=document.getElementById('saved');
  select.innerHTML=saved.length?saved.map((x,i)=>`<option value="${i}">${escapeText(x.payload.submitted_at)} · ${x.reviewed_cases}組 · ${x.sha256.slice(0,10)}</option>`).join(''):'<option>目前尚無專案保存版本</option>';
  document.getElementById('restore').disabled=saving||!saved.length;
 }
 document.getElementById('restore').addEventListener('click',()=>{
  if(saving)return;
  const chosen=saved[Number(document.getElementById('saved').value)];
  if(!chosen||!confirm('載入會取代此頁目前草稿；專案歷史版本不會刪除。無法驗證的原草稿會先另存瀏覽器備份，備份失敗就不取代。是否載入？'))return;
  try{
   const next=structuredClone(validateDraft(chosen.payload.cases,manifest));
   if(blocked){
    const backupKey=key+'.recovery.'+crypto.randomUUID();
    localStorage.setItem(backupKey,originalRaw);
    localStorage.setItem(key,JSON.stringify({manifest_sha256:sha,parent:chosen.sha256,cases:next}));
    blocked=false;storageAvailable=true;persisted=true;
   }
   cases=next;parent=chosen.sha256;error.textContent='';remember();render();
   status.textContent='已載入專案版本 '+parent+'；之後修改另存新版本。'+(persisted?'':' 此頁只在記憶體中，請勿直接關頁。');
  }catch(e){error.textContent=e.message+'；沒有取代原始草稿。';}
 });
 render();
 if(blocked)status.textContent='原草稿未覆寫，請先處理版本差異。';
 refreshSaved().catch(e=>{error.textContent+=' '+e.message+'；仍可匯出備份，不要把記憶體內容當成專案已存版本。';});
}
if(typeof document!=='undefined')mount();
