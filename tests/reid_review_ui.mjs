import assert from 'node:assert/strict';
import test from 'node:test';
import {blankCase,validateDraft,stats,payload,exportText} from '../src/bxjudge/web/reid_review.js';
const manifest={cases:[{case_id:'a'},{case_id:'b'}]};
test('no prefilled truth and unreviewed is distinct from uncertainty',()=>{
 const cases=manifest.cases.map(c=>blankCase(c.case_id));
 assert.equal(stats(cases).reviewed,0);assert.equal(cases[0].relation,'unreviewed');
 cases[0].reviewed=true;cases[0].relation='uncertain';cases[0].scope='unknown';
 assert.deepEqual(stats(cases),{reviewed:1,uncertain:1});
 assert.equal(cases[0].side_before,'unreviewed');
});
test('manifest case identity and arbitrary source fields cannot enter saved labels',()=>{
 const cases=manifest.cases.map(c=>blankCase(c.case_id));
 assert.equal(validateDraft(cases,manifest),cases);
 cases[0].case_id='elsewhere';assert.throws(()=>validateDraft(cases,manifest));
 cases[0]=blankCase('a');cases[0].source='fake';assert.throws(()=>validateDraft(cases,manifest));
});
test('save/export is a snapshot and exports valid JSON with a real newline',()=>{
 const cases=manifest.cases.map(c=>blankCase(c.case_id));
 cases[0].reviewed=true;assert.throws(()=>payload(cases,'a'.repeat(64),null));
 cases[0].relation='same';cases[0].scope='unknown';
 const p=payload(cases,'a'.repeat(64),null);cases[0].note='later unsaved change';
 assert.equal(p.cases[0].note,'');assert.equal(p.supersedes,null);
 const text=exportText(p);assert.equal(text.at(-1),'\n');assert.deepEqual(JSON.parse(text),p);
});
