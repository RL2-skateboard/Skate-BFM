"""Transactions retain the live simulation until the replacement can render."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from skate_sim.viewer import SceneWorkbench
from skate_sim.models.scene import normalize_scene_config


class WorkbenchTests(unittest.TestCase):
    def workbench(self):
        w=SceneWorkbench.__new__(SceneWorkbench)
        w.backend_name='mujoco'
        w.config=normalize_scene_config({})
        w.pending=normalize_scene_config({'style':'cruiser'})
        w.kwargs=dict(scene_config=copy.deepcopy(w.config),controller='none')
        w.pending_controller='none';w.recorder=None;w._model_xml='old XML'
        w.backend=Mock(paused=False)
        w._capture_model_xml=Mock(return_value='new XML')
        return w

    def test_compile_failure_retains_old_paused_backend(self):
        w=self.workbench();old=w.backend
        with patch('skate_sim.viewer.create_backend',side_effect=ValueError('bad compile')):
            with self.assertRaisesRegex(ValueError,'bad compile'):w.apply_scene()
        self.assertIs(w.backend,old)
        self.assertTrue(old.paused)
        old.close.assert_not_called()

    def test_renderer_failure_restores_backend_config_and_closes_replacement(self):
        w=self.workbench();old=w.backend;old_config=w.config;pending=w.pending
        replacement=Mock();w._bind_scene=Mock(side_effect=[RuntimeError('bad renderer'),None])
        with patch('skate_sim.viewer.create_backend',return_value=replacement):
            with self.assertRaisesRegex(RuntimeError,'bad renderer'):w.apply_scene()
        self.assertIs(w.backend,old)
        self.assertIs(w.config,old_config)
        self.assertIs(w.pending,pending)
        self.assertEqual(w._model_xml,'old XML')
        self.assertTrue(old.paused)
        old.close.assert_not_called()
        replacement.close.assert_called_once()

    def test_recording_rejects_stale_edit_events(self):
        w=self.workbench();w.recorder=object();before=copy.deepcopy(w.pending)
        for op,value in [('field',('board',False)),('add','rail'),('apply',None)]:
            with self.assertRaises(ValueError):w._process(op,value)
        self.assertEqual(w.pending,before)

    def test_recording_hashes_actual_selected_context(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from skate_sim.record import recording_inputs
        paths=dict(reward='reward_inference/reward_locomotion.pkl',
                   tracking='tracking_inference/zs_walking.pkl',
                   goal='goal_inference/goal_reaching.pkl')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for task,relative in paths.items():
                path=root/'model'/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(task.encode())
                rows=recording_inputs(str(root),task)
                matching=[r for r in rows if r['path']==str(path.resolve())]
                self.assertEqual(len(matching),1)
                self.assertEqual(matching[0]['sha256'],hashlib.sha256(task.encode()).hexdigest())

    def test_obstacle_ghosts_use_real_geometry_and_survive_selection(self):
        import numpy as np
        w=self.workbench();w.pending=copy.deepcopy(w.config)
        w.selected='无';w.preview=None;w.obstacle_previews={};w.server=Mock()
        w._refresh_gui=Mock(side_effect=w._show_obstacle_previews)
        handles=[]
        def mesh(*args,**kwargs):
            handle=Mock();handles.append(handle);return handle
        w.server.scene.add_mesh_simple.side_effect=mesh
        for kind in ('ramp','platform','rail'):w._process('add',kind)
        self.assertEqual(set(w.obstacle_previews),{'item_1','item_2','item_3'})
        positions=[o['pos'][0] for o in w.pending['obstacles']]
        self.assertEqual(len(set(positions)),3)
        meshes=w._obstacle_meshes(w.pending['obstacles'])
        np.testing.assert_allclose(meshes['item_1'].extents,[.7,1.1,.24],atol=1e-6)
        np.testing.assert_allclose(meshes['item_2'].bounds,[[ -.35,-.35,0],[.35,.35,.2]],atol=1e-6)
        np.testing.assert_allclose(meshes['item_3'].extents,[.8,.05,.05],atol=1e-6)
        old=list(w.obstacle_previews.values());w.selected='item_1';w._show_obstacle_previews()
        self.assertEqual(len(w.obstacle_previews),3)
        for handle in old:handle.remove.assert_not_called()
        count=w.server.scene.add_mesh_simple.call_count
        with patch.object(w,'_obstacle_meshes',side_effect=AssertionError('drag recompiled')):
            w._process('move',('item_1',[2.,3.,.4],[.70710678,0,0,.70710678]))
        self.assertEqual(w.server.scene.add_mesh_simple.call_count,count)
        self.assertEqual(w.obstacle_previews['item_1'].position,[2.,3.,.4])
        np.testing.assert_allclose(w.obstacle_previews['item_1'].wxyz,[.70710678,0,0,.70710678])
        w._process('remove',None)
        self.assertEqual(set(w.obstacle_previews),{'item_2','item_3'})
        w.config=copy.deepcopy(w.pending);w._show_obstacle_previews()
        self.assertFalse(w.obstacle_previews)
        w.pending['obstacles'][0]['kind']='rail';w._show_obstacle_previews()
        self.assertEqual(set(w.obstacle_previews),{'item_2'})

    def test_ghost_pose_equivalence_after_normalization_and_quaternion_sign(self):
        w=self.workbench();w.server=Mock();w.selected='item_1';w.obstacle_previews={}
        w.config=normalize_scene_config(dict(obstacles=[dict(id='item_1',kind='ramp',
            pos=[1.,2.,0.],quat=[1.,2.,3.,4.])]))
        w.pending=copy.deepcopy(w.config)
        w._show_obstacle_previews()
        self.assertFalse(w.obstacle_previews)
        w.pending['obstacles'][0]['quat']=[-q for q in w.pending['obstacles'][0]['quat']]
        w._show_obstacle_previews()
        self.assertFalse(w.obstacle_previews)
        w.pending['obstacles'][0]['pos'][0]+=.01
        w._show_obstacle_previews()
        self.assertEqual(set(w.obstacle_previews),{'item_1'})

    def test_scene_dirty_uses_same_physical_poses_and_preserves_order(self):
        w=self.workbench()
        pose=dict(pos=[0.,1.,2.],quat=[1.,2.,3.,4.])
        w.config=normalize_scene_config(dict(robot_pose=pose,board_pose=pose,
            obstacles=[dict(id='a',kind='ramp',**pose),dict(id='b',kind='rail',**pose)]))
        w.pending=copy.deepcopy(w.config)
        for key in ('robot_pose','board_pose'):
            w.pending[key]['quat']=[-q for q in w.pending[key]['quat']]
        for o in w.pending['obstacles']:o['quat']=[-q for q in o['quat']]
        self.assertFalse(w._scene_dirty())
        # The recording guard must accept equivalent poses and proceed to input capture.
        with patch('skate_sim.viewer.recording_inputs',side_effect=RuntimeError('capture reached')):
            with self.assertRaisesRegex(RuntimeError,'capture reached'):w.start_recording('unused')
        for key in ('robot_pose','board_pose'):
            w.pending[key]['pos'][0]+=.01
            self.assertTrue(w._scene_dirty())
            w.pending[key]['pos'][0]-=.01
        w.pending['obstacles'][0]['pos'][0]+=.01
        self.assertTrue(w._scene_dirty())
        w.pending['obstacles'][0]['pos'][0]-=.01
        self.assertFalse(w._scene_dirty())
        w.pending_controller='pose';self.assertTrue(w._scene_dirty())
        w.pending_controller='none';w.pending['obstacles'].reverse()
        self.assertTrue(w._scene_dirty())
