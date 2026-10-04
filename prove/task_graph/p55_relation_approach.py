"""Public EEF-only relation continuation after the frozen P53 approach waypoint."""
import numpy as np
from scipy.spatial.transform import Rotation

from .p52_training_teacher import norm_limit


class PublicRelationApproach:
    """Continue the public ref41->ref57 EEF axis; no object or goal observation."""

    def __init__(self, reference41, reference57):
        p41 = np.asarray(reference41['position'], dtype=float)
        p57 = np.asarray(reference57['position'], dtype=float)
        displacement = p57-p41
        length = float(np.linalg.norm(displacement))
        if p41.shape != (3,) or p57.shape != (3,) or not np.isfinite(length) or length <= 1e-9:
            raise ValueError('Invalid public EEF approach axis')
        self.axis = displacement/length
        self.origin = p57
        self.orientation = Rotation.from_quat(reference57['quaternion_xyzw']).as_matrix()

    def command(self, position, quaternion):
        position = np.asarray(position, dtype=float)
        current = Rotation.from_quat(quaternion).as_matrix()
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Invalid public EEF position')
        error = position-self.origin
        axial_progress = float(error @ self.axis)
        lateral = error-axial_progress*self.axis
        # Use the original PublicPosePath 0.01 m command-request limit; physical
        # displacement is determined by later closed-loop observations.
        dp = self.axis*.01-lateral
        dr = Rotation.from_matrix(self.orientation@current.T).as_rotvec()
        action = np.r_[norm_limit(dp,.01)/.05, norm_limit(dr,.05)/.5, 1.]
        if not np.isfinite(action).all():
            raise ValueError('Nonfinite public approach command')
        return action, dict(direction=self.axis.tolist(), axial_progress=axial_progress,
            lateral_error=lateral.tolist(), lateral_error_norm=float(np.linalg.norm(lateral)),
            public_eef_position=position.tolist(), public_eef_quaternion_xyzw=np.asarray(quaternion).tolist(),
            reference_orientation_xyzw=Rotation.from_matrix(self.orientation).as_quat().tolist(),
            position_step_request=dp.tolist(), rotation_error=dr.tolist())
