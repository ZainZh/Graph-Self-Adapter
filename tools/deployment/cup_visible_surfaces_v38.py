"""Multiple visible surfaces of one hollow SAM cup; bounded by initial visible geometry."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from cup_probe_lift_v37_runtime import SharedCupExecutor as BaseExecutor,SharedSAMVision
from cup_v25_visible_support_geometry import dominant_visible_surface as single_surface
DIAMETER_BOUND=None

def observed_surfaces(points):
    global DIAMETER_BOUND
    try:return single_surface(points)
    except SafetyError as error:
        if str(error)!='current cup surface disconnected/ambiguous':raise
    cloud=np.asarray(points,float)
    if DIAMETER_BOUND is None:raise SafetyError('initial visible cup size prior missing')
    pairs=cKDTree(cloud).query_pairs(.006,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(cloud),len(cloud)))
    _,labels=connected_components(graph,directed=False);sizes=np.bincount(labels)
    large=np.where(sizes>=200)[0]
    if len(large)<2 or len(large)>4:raise SafetyError('cup has unsupported visible surface fragmentation')
    selected=cloud[np.isin(labels,large)];coverage=len(selected)/len(cloud)
    extent=float(np.linalg.norm(np.ptp(selected,axis=0)))
    if coverage<.70 or extent>DIAMETER_BOUND:raise SafetyError('multiple cup surfaces lack bounded instance geometry')
    report={'input_points':len(cloud),'dominant_points':len(selected),'component_count':len(sizes),'dominant_fraction':coverage,'neighbor_radius_m':.006,'representation':'multiple visible components of one actual SAM cup instance','component_sizes':sizes[large].tolist(),'current_visible_bbox_diagonal_m':extent,'initial_visible_cup_bound_m':DIAMETER_BOUND,'minimum_component_points':200,'minimum_total_coverage':.70,'not_connected_contact_or_hidden_mesh':True}
    return selected,report

def install_surface_queries():
    import cup_v25_visible_support_geometry,cup_v25_full_support_envelope,cup_v25_full_envelope_runtime,cup_v25_visible_support_runtime
    for module in (cup_v25_visible_support_geometry,cup_v25_full_support_envelope,cup_v25_full_envelope_runtime,cup_v25_visible_support_runtime):module.dominant_visible_surface=observed_surfaces

class HollowCupExecutor(BaseExecutor):
    def start(self):
        global DIAMETER_BOUND
        super().start();measurements={}
        for role in ('white_mug','red_mug'):
            cloud=self.vision.cloud(role)
            if len(cloud)<200:raise SafetyError('initial actual cup surface insufficient for extent prior')
            measurements[role]=float(np.linalg.norm(np.ptp(cloud,axis=0)))
        DIAMETER_BOUND=max(measurements.values())+.010
        write(self.output/'INITIAL_VISIBLE_CUP_EXTENTS.json',{'source':'actual initial SAM RGBD, no object CAD/truth','visible_bbox_diagonal_m':measurements,'max_visible_diagonal_allowance_m':DIAMETER_BOUND,'development_tolerance_m':.010,'not_hidden_full_shape':True})
    def segment_goals(self,stage,pose):
        if stage['cycle_id']=='white_direct' and stage['phase']=='Lower':
            summaries={}
            for role in ('white_mug','red_mug'):
                _,report=observed_surfaces(self.vision.cloud(role));summaries[role]=report
            write(self.output/'CURRENT_CUP_SURFACE_VALIDATION.json',summaries)
        return super().segment_goals(stage,pose)

install_surface_queries()
SharedCupExecutor=HollowCupExecutor
