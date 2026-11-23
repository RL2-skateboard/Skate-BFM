import numpy as np
from skate_bfm.integration.observations import bfm0_privileged_state


def test_physical_body_observations_include_virtual_head_velocity():
    pos=np.zeros((30,3));quat=np.zeros((30,4));quat[:,0]=1
    vel=np.zeros((30,3));omega=np.zeros((30,3));omega[15,0]=2
    result=bfm0_privileged_state(pos,quat,vel,omega)
    assert result.shape==(463,)
    np.testing.assert_allclose(result[277+90:277+93],[0.,-.7,0.])
