"""Strict continuation of a saved run onto an exact-prefix extended schedule."""
from distill_runtime import read,digest

def restore(args,data,teacher,model,optimizer,torch):
    assert args.resume_parent_root and not args.paired_orders
    root=args.resume_parent_root
    old=read(root/'prepared/dataset.json.gz');targets=read(root/'teacher/teacher.json');result=read(root/'student/result.json')
    assert result['status']=='completed'
    state=torch.load(args.resume_checkpoint,map_location='cpu',weights_only=False)
    step=state['scheduler']['completed_updates']
    assert step==500 and args.steps==args.schedule_steps==1800
    assert state['scheduler']['steps']==args.schedule_steps
    assert state['scheduler']['schedule']==args.schedule and state['scheduler']['peak_lr']==args.learning_rate
    assert state['config']['student_order']==args.student_order=='document_first'
    assert state['dataset_sha256']==digest(root/'prepared/dataset.json.gz')==result['dataset_sha256']
    assert state['teacher_sha256']==digest(root/'teacher/teacher.json')==result['teacher_sha256']
    assert state['training_order_ids']==[g['id'] for g in old['train']]
    assert data['train'][:len(old['train'])]==old['train']
    for section in ('validation','benchmark','reserved_benchmark'):
        assert data[section]==old[section]
    assert not teacher.get('stream'), 'Continuation requires complete verified teacher cache'
    for section in ('train','validation','benchmark','reserved_benchmark'):
        assert all(teacher['scores'][section][k]==v for k,v in targets['scores'][section].items())
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    assert len(optimizer.state)>0
    assert all(int(v['step'].item() if hasattr(v['step'],'item') else v['step'])==step for v in optimizer.state.values())
    torch.set_rng_state(state['rng']);torch.npu.set_rng_state(state['npu_rng'])
    del state
    return step,result
