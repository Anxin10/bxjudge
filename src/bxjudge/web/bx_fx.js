// BXJudge workbench motion layer. Everything here is decoration: with motion off
// (toggle or prefers-reduced-motion) every value is set immediately and stays readable.
(function(){
 const KEY='bxjudge.fx';
 function stored(){try{return localStorage.getItem(KEY);}catch(e){return null;}}
 function store(v){try{localStorage.setItem(KEY,v);}catch(e){}}
 const reduced=()=>typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
 const on=()=>document.body.classList.contains('fx-on');
 function apply(enabled){
  document.body.classList.toggle('fx-on',enabled);
  for(const b of document.querySelectorAll('.fx-toggle')){
   b.setAttribute('aria-pressed',String(enabled));b.textContent=enabled?'動效：開':'動效：關';
  }
 }
 function count(el,value,digits,suffix){
  digits=digits??2;suffix=suffix??'';
  const done=()=>{el.textContent=Number(value).toFixed(digits)+suffix;};
  if(!on()||!Number.isFinite(value))return done();
  const start=performance.now(),dur=700,from=0;
  // Overshoot slightly then settle, like a value being computed.
  function frame(now){
   const p=Math.min(1,(now-start)/dur),e=1-Math.pow(1-p,3),wobble=p<1?Math.sin(p*9)*(1-p)*0.06:0;
   el.textContent=(from+(value-from)*e+wobble*Math.abs(value||1)).toFixed(digits)+suffix;
   if(p<1)requestAnimationFrame(frame);else done();
  }
  requestAnimationFrame(frame);
 }
 function glitch(el){
  if(!el||!on())return;
  el.classList.remove('glitch');void el.offsetWidth;el.classList.add('glitch');
  setTimeout(()=>el.classList.remove('glitch'),450);
 }
 function reveal(list){
  const items=[...list.children];
  if(!on()){items.forEach(i=>i.style.opacity='');return;}
  items.forEach((item,i)=>{item.style.opacity='0';setTimeout(()=>{item.style.transition='opacity .25s';item.style.opacity='';},90*i+120);});
 }
 function flowDots(svg){
  for(const old of svg.querySelectorAll('.dot'))old.remove();
  let i=0;
  for(const path of svg.querySelectorAll('path.edge')){
   const dot=document.createElementNS('http://www.w3.org/2000/svg','circle');
   dot.setAttribute('r','4');dot.setAttribute('class','dot');
   dot.style.offsetPath=`path("${path.getAttribute('d')}")`;
   dot.style.setProperty('--dur',(path.classList.contains('loop')?5:2.4)+'s');
   dot.style.setProperty('--delay',(i++*0.35)+'s');
   if(path.classList.contains('loop'))dot.style.fill='var(--coral)';
   path.after(dot);
  }
 }
 function init(){
  const pref=stored();
  apply(pref?pref==='on':!reduced());
  document.addEventListener('click',e=>{
   const b=e.target.closest&&e.target.closest('.fx-toggle');if(!b)return;
   const next=!on();store(next?'on':'off');apply(next);
   document.dispatchEvent(new CustomEvent('bxfx',{detail:{on:next}}));
  });
 }
 window.BXFX={init,count,glitch,reveal,flowDots,on};
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();
