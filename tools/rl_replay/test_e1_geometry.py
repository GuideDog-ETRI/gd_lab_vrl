"""CPU real-mesh regression; Isaac base/config classes are minimal stubs, no app."""
import os,sys,ast,types,unittest,__future__
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
import torch
import e1_runner
ROOT=Path(__file__).resolve().parents[2]

def extract(path,names,ns):
    tree=ast.parse((ROOT/path).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),ns)

class GeometryTest(unittest.TestCase):
    def test_v21_gap20_real_mesh_metadata_and_lookup(self):
        N=types.SimpleNamespace
        ns={'np':np,'trimesh':trimesh,'torch':torch,'configclass':lambda c:c,'SubTerrainBaseCfg':object}
        extract('src/gd_lab/mdp/platform_gap_mesh.py',['build_boarding_mesh'],ns)
        extract('src/gd_lab/mdp/platform_gap_metadata.py',['boarding_tile_metadata'],ns)
        extract('src/gd_lab/mdp/terrains/platform_gap.py',['platform_gap_terrain','PlatformGapTerrainCfg'],ns)
        # Stub only Isaac placement. The actual metadata subclass/method is run.
        class BaseGenerator:
            def __init__(self,cfg,device='cpu'):self.cfg=cfg
            def _add_sub_terrain(self,mesh,origin,row,col,sub):return mesh,origin
        ns['TerrainGenerator']=BaseGenerator
        extract('src/gd_lab/mdp/terrains/gap_metadata_generator.py',['GapMetadataTerrainGenerator'],ns)
        extract('src/gd_lab/mdp/gap_stair_v2.py',['set_gap_terrain_columns'],ns)
        extract('src/gd_lab/mdp/terrain_families.py',['family_column_masks'],ns)
        ns['_CACHE_ATTR']='_gd_family_cols'
        extract('src/gd_lab/mdp/platform_gap_finetune.py',['GapTileGeometry'],ns)
        sub=ns['PlatformGapTerrainCfg']();sub.size=(8.,8.);sub.proportion=5.
        # Exactly the v21 student's v2 width override; no metadata initially.
        source=ast.parse((ROOT/'src/gd_lab/mdp/gap_stair_v2.py').read_text())
        width=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='V2_GAP_WIDTH_RANGE' for t in n.targets))
        sub.gap_width_range=width
        cfg=N(num_rows=15,num_cols=15,size=(8.,8.),seed=42,sub_terrains={**{f'other{i}':N(proportion=1.) for i in range(10)},'platform_gap':sub},class_type=BaseGenerator)
        ec=N(scene=N(terrain=N(terrain_generator=cfg)),curriculum=N(terrain_levels=object()))
        ns['set_gap_terrain_columns'](ec,20)
        self.assertEqual(cfg.num_cols,30)
        b=N(device='cpu',scene=N(terrain=N(cfg=N(terrain_generator=cfg),terrain_levels=torch.tensor([0,7,14]),terrain_types=torch.tensor([10,20,29]))))
        with self.assertRaisesRegex(ValueError,'requires GapMetadataTerrainGenerator'):ns['GapTileGeometry'](b)
        module=N(GapMetadataTerrainGenerator=ns['GapMetadataTerrainGenerator'])
        with patch.dict(sys.modules,{'gd_lab.mdp.terrains.gap_metadata_generator':module}):e1_runner.configure(ec)
        self.assertIs(cfg.class_type,ns['GapMetadataTerrainGenerator'])
        gen=cfg.class_type(cfg,device='cpu');rng=np.random.default_rng(42)
        errors=[]
        for row in range(15):
            for col in range(10,30):
                # Same curriculum stratum + jitter contract as TerrainGenerator.
                difficulty=(row+rng.random())/15
                np.random.seed(row*30+col)
                meshes,origin=ns['platform_gap_terrain'](difficulty,sub)
                mesh=trimesh.util.concatenate(meshes);mesh.apply_translation([-4,-4,0]);origin=origin-np.array([4,4,0])
                before=mesh.vertices.copy();state=np.random.get_state()
                gen._add_sub_terrain(mesh,origin,row,col,sub)
                np.testing.assert_array_equal(mesh.vertices,before)
                after=np.random.get_state();np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])
                tile=cfg.gap_tile_metadata[row][col];w=width[0]+difficulty*(width[1]-width[0])
                expected=np.array([[-1.5-w/2,-1.5+w/2],[1.5-w/2,1.5+w/2]])
                errors.append(float(np.max(np.abs(np.array(tile['slots'])-expected))))
                self.assertLess(errors[-1],1e-12)
                self.assertEqual(tile['lower_decks'],[min(tile['decks'][:2]),min(tile['decks'][1:])])
        geometry=ns['GapTileGeometry'](b);slots,lower=geometry.for_envs(b)
        for i,(row,col) in enumerate(zip([0,7,14],[10,20,29])):
            torch.testing.assert_close(slots[i],torch.tensor(cfg.gap_tile_metadata[row][col]['slots']))
        # Recorder's world-frame conversion: origin translation exactly once.
        b.scene.env_origins=torch.tensor([[10.,0,.1],[20.,0,.2],[30.,0,.3]])
        recorder=object.__new__(e1_runner.Recorder);recorder.base=b;recorder.geometry=geometry
        near,far,deck=recorder.edges()
        torch.testing.assert_close(near,slots[:,1,0]+b.scene.env_origins[:,0])
        torch.testing.assert_close(far,slots[:,1,1]+b.scene.env_origins[:,0])
        torch.testing.assert_close(deck,lower[:,1]+b.scene.env_origins[:,2])
        print('PASS v21 width',width,'20/30 columns; 300 real gap meshes; max edge error',max(errors))
    def test_configure_before_environment_creation(self):
        s=(ROOT/'scripts/gast/train_student_live.py').read_text()
        self.assertLess(s.index('configure(env_cfg)'),s.index('env = gym.make'))

if __name__=='__main__':torch.set_num_threads(1);unittest.main(verbosity=2)
