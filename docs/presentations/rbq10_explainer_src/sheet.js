const sharp=require('sharp'),fs=require('fs');
(async()=>{const f=fs.readdirSync('preview').filter(x=>x.endsWith('.jpg')).sort();
for(let i=0;i<f.length;i+=4){const comps=[];for(let j=0;j<4&&i+j<f.length;j++){comps.push({input:await sharp('preview/'+f[i+j]).resize(960,540).toBuffer(),left:(j%2)*965,top:Math.floor(j/2)*545});}
await sharp({create:{width:1925,height:1085,channels:3,background:'#888'}}).composite(comps).jpeg({quality:85}).toFile('sheet-'+(i/4+1)+'.jpg');}
console.log('ok')})();
