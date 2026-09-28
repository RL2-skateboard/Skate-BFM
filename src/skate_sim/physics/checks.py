"""Stage 03 physical regression. Fixed engineering thresholds, not real accuracy."""
from dataclasses import asdict
import numpy as np
from ..config import SimConfig
from ..backends.mujoco import MujocoBackend
from ..models.board import board_styles
from .dynamics import experiment_config


def rollout(style, model, experiment, seconds, dt=.002, sign=1, loss=1):
    cfg=asdict(experiment_config(model,experiment))
    cfg['lean_target'] *= sign
    cfg['wheel_spin_damping'] *= loss
    cfg['bearing_torque'] *= loss
    b=MujocoBackend(scene='board',style=style,experiment=experiment,dynamics=cfg,
                    config=SimConfig(physics_dt=dt))
    initial=b.diagnostics()['mechanical_energy_J']
    peak_excess=0.; minimum_contacts=10000; maximum_contacts=0
    initial_displacement=np.linalg.norm([b.data.joint(p+'_truck_tilt').qpos[0] for p in ('front','rear')])
    ground=b.model.geom('ground').id
    try:
        for i in range(round(seconds/dt)):
            b.step()
            if i % 20 == 0:
                assert np.isfinite(b.data.qpos).all() and np.isfinite(b.data.qvel).all(), 'nonfinite state'
                d=b.diagnostics()
                peak_excess=max(peak_excess,d['mechanical_energy_J']-initial-b.external_work)
                ground_contacts=sum(ground in (c.geom1,c.geom2) for c in b.data.contact)
                minimum_contacts=min(minimum_contacts,ground_contacts)
                maximum_contacts=max(maximum_contacts,ground_contacts)
        d=b.diagnostics()
        return dict(position=b.data.joint('deck_free').qpos[:3].tolist(),
                    velocity=b.data.joint('deck_free').qvel[:3].tolist(),
                    roll=d['deck_roll_rad'], steer=d['steer_rad'],
                    spring_norm=float(np.linalg.norm([b.data.joint(p+'_truck_tilt').qpos[0] for p in ('front','rear')])),
                    initial_spring_norm=float(initial_displacement),
                    energy_excess_J=peak_excess, min_contacts=int(minimum_contacts),max_contacts=int(maximum_contacts))
    finally: b.close()


def check_dynamics(styles):
    results={}
    for style in board_styles(styles):
        for model in ('truck','reduced'):
            key=style+'/'+model
            evidence={}
            try:
                coast=rollout(style,model,'coast',60); evidence['coast_60s']=coast
                assert abs(coast['position'][1]) < .02, 'straight coast lateral drift > 2 cm'
                assert coast['energy_excess_J'] < .05, 'unexplained energy gain > .05 J'
                release=rollout(style,model,'release',3); evidence['release']=release
                assert release['spring_norm'] < release['initial_spring_norm']*.2, 'release did not decay by 80%'
                left=rollout(style,model,'turn',2); right=rollout(style,model,'turn',2,sign=-1)
                evidence['turn']=[left,right]
                assert left['roll'] > .005 and right['roll'] < -.005, 'deck did not lean in commanded direction'
                assert left['steer']['front'] < -.005 and left['steer']['rear'] > .005, 'axles did not counter-steer relative to deck heading'
                y1,y2=left['position'][1],right['position'][1]
                assert y1 < -.01 and y2 > .01, 'incorrect turn sign or negligible turning'
                assert abs(y1+y2)/max(abs(y1),abs(y2)) < .1, 'mirror error > 10%'
                half=rollout(style,model,'turn',2,dt=.001); evidence['dt_half']=half
                assert np.linalg.norm(np.array(left['position'])-half['position'])/np.linalg.norm(left['position']) < .05, 'dt position error > 5%'
                drop=rollout(style,model,'drop',2); evidence['drop']=drop
                assert drop['min_contacts']==0 and drop['max_contacts']>=4, 'no separation/recontact'
                nominal=rollout(style,model,'coast',2); lossy=rollout(style,model,'coast',2,loss=5)
                evidence['loss_distance']=[nominal['position'][0],lossy['position'][0]]
                assert lossy['position'][0] < nominal['position'][0], 'loss parameter ineffective'
                results[key]=dict(status='PASS',evidence=evidence)
            except Exception as exc:
                results[key]=dict(status='FAIL',error=str(exc),evidence=evidence)
    return results
