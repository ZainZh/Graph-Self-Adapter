import numpy as np
from tools.task_graph.collect_libero_p52_variants import conditions, VariantController
from prove.task_graph.libero_precision_p51 import MixedController


def test_fixed_conditions_have_prespecified_disjoint_splits():
    relation=dict(book_collision_halfsize=[.01434,.05519,.06724],back_halfsize=[.02775,.06216,.05])
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    rows=conditions(relation,R)
    assert len(rows)==24
    assert sum(r['split']=='train' for r in rows)==18
    assert sum(r['split']=='validation' for r in rows)==6
    assert len({tuple(r['grasp_book_offset']+r['mouth_offset']) for r in rows})==24
    for i,r in enumerate(rows):
        assert r['condition_seed']==520000+i
        assert r['job_id']==f'job_{i:02d}'
        np.testing.assert_allclose(r['grasp_world_offset'],R@r['grasp_book_offset'])
        assert (r['split']=='validation')==(r['quadrant_index']==r['grasp_index']%4)


def test_grasp_reference_shift_is_scoped_and_restored(monkeypatch):
    skill=object.__new__(VariantController)
    skill.offset=np.array([.001,-.002,.003]);skill.nominal_reference=np.eye(4)
    observations=[]
    def call(self,payload):
        observations.append(self.nominal_reference.copy())
        return np.zeros(7)
    monkeypatch.setattr(MixedController,'__call__',call)
    skill({'stage_id':'s3'});skill({'stage_id':'s6'})
    np.testing.assert_allclose(observations[0][:3,3],skill.offset)
    np.testing.assert_array_equal(observations[1],np.eye(4))
    np.testing.assert_array_equal(skill.nominal_reference,np.eye(4))
