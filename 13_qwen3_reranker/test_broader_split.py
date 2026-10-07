import unittest
from broader_split import clean_pool, sample_group, SPECS, strict_key

class SplitTests(unittest.TestCase):
 def row(self):
  return dict(id='nq/1',source='nq',query='Which planet?',pos=['Mars',' MARS ','red planet'],neg=['Mars','Venus','Earth'],original_positive_count=3,original_negative_count=3)
 def test_upstream_membership_and_holdouts(self):
  row=self.row();reg={'train':{'which planet?':'q1'}}
  clean,counts=clean_pool([row],'nq',reg,{strict_key('Which planet?')})
  self.assertEqual(clean,[])
  clean,counts=clean_pool([row],'nq',{'train':{}},set())
  self.assertEqual(counts['not_verified_upstream_train'],1)
  clean,_=clean_pool([row,row],'nq',reg,set())
  self.assertEqual(len(clean),1);self.assertEqual(clean[0]['upstream_query_id'],'q1')
  self.assertEqual(clean[0]['pos'],['Mars','red planet']);self.assertNotIn('Mars',clean[0]['neg'])
 def test_no_supplementation_and_no_known_positive_as_negative(self):
  rows,_=clean_pool([self.row()],'nq',None,set());r=rows[0]
  self.assertIsNone(sample_group(r,lambda r,d: True,4))
  r['neg'] += [f'Other {i}' for i in range(20)]
  g=sample_group(r,lambda r,d: True,4)
  self.assertEqual(len(g['documents']),8)
  self.assertTrue(all(d in r['neg'] for d in g['documents'][1:]))
  self.assertEqual(g['supplied_labels'],[1]+[0]*7)
  self.assertEqual(g,sample_group(r,lambda r,d: True,4))
  self.assertIsNone(sample_group(r,lambda r,d: False,4))
 def test_capped_raw_pools_rejected(self):
  for field in ['pos','neg']:
   r=self.row();r[field]=r[field][:-1]
   with self.assertRaisesRegex(ValueError,'pool was capped'):
    clean_pool([r],'nq',None,set())
 def test_language_and_pilot_quotas(self):
  for lang in ['en','zh']:
   self.assertEqual(sum(n for l,n,v,i in SPECS.values() if l==lang),3200)
   self.assertEqual(sum(v for l,n,v,i in SPECS.values() if l==lang),160)
  self.assertTrue(all(n%4==0 for l,n,v,i in SPECS.values()))
if __name__=='__main__':unittest.main()
