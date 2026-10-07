// CPU-only synthetic/live state tests. No browser, WebGL or display.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(__dirname+'/viewer.html','utf8');
const nodes=new Map();
function node(){return {value:'0',hidden:false,disabled:false,dataset:{},style:{},clientWidth:600,
  parentElement:{},textContent:'',innerHTML:'',children:[],append(...c){this.children.push(...c)},replaceChildren(...c){this.children=c},
  querySelectorAll(){return []},addEventListener(){},getContext(){return new Proxy({}, {get:()=>()=>{}})}}}
const context={console,URL,URLSearchParams,Option:function(){},location:{protocol:'file:',href:'file:///tmp/viewer.html',search:'',hash:''},
 document:{getElementById(id){if(!nodes.has(id))nodes.set(id,node());return nodes.get(id)},
 createElement:node,addEventListener(){}},window:{devicePixelRatio:1},
 performance:{now:()=>0},requestAnimationFrame:()=>1,cancelAnimationFrame(){},
 setTimeout:()=>1,clearTimeout(){},fetch:()=>new Promise(()=>{}),getComputedStyle:()=>({getPropertyValue:()=>''}),alert(){}};
vm.createContext(context);
for(const m of html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g))vm.runInContext(m[1],context);
vm.runInContext(`
draw=()=>{};
D={meta:{steps:2,dt:.01},envs:[]};E=null;t=0;
setPlaying(true);if(!playing)throw Error("play not started");
startLive("a.live.ndjson");if(playing||D!==null)throw Error("live transition kept playback");
tick(100); // must not dereference null
D={meta:{gamma:.9,lam:.95,term_names:["a","b"]},envs:[{reward:[1,2],value:[3,4],done:[0,0],terms:[[.4,.6],[.8,1.2]]}]};
finishProvisional();
if(Math.abs(D.envs[0].return[0]-6.04)>1e-8)throw Error("provisional bootstrap mismatch");
if(Math.abs(D.envs[0].advantage[0]-2.968)>1e-8)throw Error("provisional GAE mismatch");
E={scan:[[[0,0,0]]],terrain_obs:[[0,1]],terrain_scan:[[[null,0,0]]],terrain_scan_z:[1]};
D.meta.terrain_coordinates="capture_aligned";t=0;if(teacherCells().length)throw Error("null coordinate rendered");
let prior=generation;beginLoad();if(generation!==prior+1)throw Error("generation not invalidated");
`,context);
assert(!html.includes('PPO는 이 행동의 확률'));
vm.runInContext(`
D={meta:{steps:1,dt:.01,task:"<unsafe>",driver:"student",term_names:["small","largest","middle","last"]}};
E={value:[0],return:[0],advantage:[0],delta:[0],reward:[0],command:[[.8,0,0]],foot_contact:[[1,0,1,0]],
 latent_err:[.12],student_fresh:[1],teacher_action:[[2,4]],action:[[1,2]],term_returns:[[1,-10,5,0]]};t=0;
drawInfo();
renderRecordings([{name:"a",meta:{},live:true,finished:false},{name:"b",meta:{},live:true,finished:true},{name:"c",meta:{}}]);
`,context);
function contents(n){return [n.textContent,...n.children.map(contents)].join(' ')}
for(const text of ['학생 latent MSE','학생 프레임 신선도','교사 행동 평균 |차|','1.500000','미래 보상 상위 3항목','largest','middle','학생'])
 assert(contents(nodes.get('info')).includes(text),text);
for(const text of ['이름','상태','● 실시간','스트림(완료)','완료'])assert(contents(nodes.get('recList')).includes(text),text);
assert(!contents(nodes.get('info')).includes('last:'),'only top 3 terms');
console.log('viewer synthetic/live transition, provisional math, null geometry, generation: PASS');
