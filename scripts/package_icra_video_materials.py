"""Audit and assemble the requested ICRA media without relabeling failures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from build_paper_sim_video720p import validate

parser=argparse.ArgumentParser(description=__doc__)
for name in ['fruit-orange','fruit-banana','drawer-panel','drawer-handle','cup-blocked']:
    parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--destination',type=Path,default=Path('/raid/home/than/icra_video_materials'))


def main():
    args=parser.parse_args();dst=args.destination
    dst.mkdir(parents=True,exist_ok=True)
    for child in ['metadata','images','trajectories']: (dst/child).mkdir(exist_ok=True)
    entries=[];initial_fruits=[];initial_drawers=[]

    def copy(source,target):
        if target.exists():
            if hashlib.sha256(source.read_bytes()).digest()!=hashlib.sha256(target.read_bytes()).digest():
                raise FileExistsError(target)
        else:shutil.copy2(source,target)

    for key,folder,kind in [
        ('two_fruit_orange_first_physical_success',args.fruit_orange,'fruit'),
        ('two_fruit_banana_first_physical_success',args.fruit_banana,'fruit'),
        ('drawer_close_panel_push_success',args.drawer_panel,'drawer'),
        ('drawer_close_handle_grasp_success',args.drawer_handle,'drawer'),
    ]:
        report=json.loads((folder/'report.json').read_text())
        assert report['success'],folder
        with np.load(folder/'episode.npz') as episode:
            if kind=='fruit':
                assert all(report['bilateral_contact_verified'].values())
                assert all(report['physical_lift_verified'].values())
                assert min(report['final_plate_contact_N'].values())>.05
                assert report['final_home_joint_max_error_rad']<.015
                transport=np.array([str(p).endswith(':transfer') for p in episode['phase']])
                assert transport.any()
                assert (episode['finger_contact_forces'][transport]>.1).all(1).mean()>.95
                initial_fruits.append(episode['state'][0].copy())
            else:
                initial_drawers.append(episode['oracle'][0].copy())
                assert max(episode['drawer_opening'][-10:])<.01
                mask=episode['phase']=='close'
                contact=np.linalg.norm(episode['contact_forces'][mask],axis=-1).max(axis=1)
                assert (contact.max(axis=1)>.5).sum()>10
                if report['method']=='handle_grasp':
                    gap=episode['oracle'][mask,7:9]
                    valid=(gap.min(1)>=.0015)&(gap.max(1)<=.012)&(abs(gap[:,0]-gap[:,1])<=.004)
                    assert valid.mean()>.95
                    assert (contact[:,:2]>.5).all(1).mean()>.9
        video=validate(folder/'demo.mp4',20,report['frames'])
        entries.append(dict(file=key+'.mp4',source=str(folder.resolve()),validation=video,report=report))
    np.testing.assert_allclose(initial_fruits[0],initial_fruits[1],atol=1e-6,rtol=0)
    np.testing.assert_allclose(initial_drawers[0],initial_drawers[1],atol=1e-6,rtol=0)
    failed=json.loads((args.cup_blocked/'report.json').read_text())
    assert failed['blocked_candidate_verified'] and failed['peak_drawer_contact_N']>.5
    assert failed['cup_displacement_m']<.02
    video=validate(args.cup_blocked/'blocked_top_grasp.mp4',20,failed['frames'])
    entries.append(dict(file='cup_below_drawer_top_grasp_blocked.mp4',source=str(args.cup_blocked.resolve()),validation=video,report=failed))

    for entry in entries:
        folder=Path(entry['source']);key=Path(entry['file']).stem
        blocked=key=='cup_below_drawer_top_grasp_blocked'
        copy(folder/('blocked_top_grasp.mp4' if blocked else 'demo.mp4'),dst/entry['file'])
        copy(folder/'report.json',dst/'metadata'/(key+'.json'))
        copy(folder/('candidate.npz' if blocked else 'episode.npz'),dst/'trajectories'/(key+'.npz'))
    copy(args.cup_blocked/'blocked_candidate.png',dst/'images'/'cup_top_grasp_blocked_720p.png')
    copy(args.cup_blocked/'first_contact.png',dst/'images'/'cup_top_grasp_first_contact_720p.png')
    copy(args.drawer_panel/'runtime_bindings.json',dst/'metadata'/'drawer_runtime_bindings.json')
    contract=dict(schema='icra_demo_contract_v1',
        provenance='Manually authored task semantics from the user request; execution bindings exported from the simulator. Not automatic video-model output.',
        entities=[dict(name='drawer',role='manipulated_object'),
                  dict(name='front_panel',role='interaction_part',part_of='drawer'),
                  dict(name='handle',role='interaction_part',part_of='drawer')],
        goal=dict(predicate='CLOSED',subject='drawer',max_opening_m=.01,settle_duration_s=.5),
        alternatives=[dict(method='panel_push',part='front_panel',requires_bilateral_grasp=False),
                      dict(method='handle_grasp',part='handle',requires_bilateral_grasp=True)],
        bindings_file='drawer_runtime_bindings.json',
        part_colliders=dict(front_panel='/World/envs/env_0/Cabinet/UpperDrawer/Front',handle='/World/envs/env_0/Cabinet/UpperDrawer/HandleBar'))
    (dst/'metadata'/'drawer_task_contract.json').write_text(json.dumps(contract,indent=2)+'\n')

    old=Path('/raid/home/than/paper_sim_video720p/know_demo')
    for key in ['cup_in_drawer_success','cup_to_cabinet_success','microwave_close_upper_contact_success','microwave_close_lower_contact_success']:
        source=old/(key+'.mp4');validation=validate(source,20)
        copy(source,dst/source.name)
        entries.append(dict(file=source.name,source=str(source),validation=validation))
    (dst/'metadata'/'manifest.json').write_text(json.dumps(dict(
        videos=entries,fruit_initial_states_identical=True,drawer_initial_states_identical=True,
        fruit_target_plate='purple',excluded='All kinematically attached fruit demos and unverified trials'),indent=2)+'\n')
    video_lines='\n'.join(f"- [{e['file']}]({e['file']}) — {e['validation']['duration_s']:.2f} 秒" for e in entries)
    readme=f'''# ICRA 视频素材

共 {len(entries)} 条 MP4，均为 1280×720、20 fps。原有文件未修改。

{video_lines}

## 验证与使用范围

- 水果：同一初始状态、同一个紫色目标盘，橘子先放／香蕉先放。水果为动态刚体，靠双指接触搬运；没有物体附着、位姿改写或搬运辅助力。检查了抬起、搬运时的持续双指接触、释放后的盘面接触与稳定性；结束时机械臂回默认关节角。
- 抽屉：同一上层抽屉、同一初始状态（打开 22 cm），分别推面板和握把手关闭。握把手版验证了关闭全过程的双指接触与非零夹爪间隙。撤开后剩余开度小于 1 cm。
- 杯子失败候选：新补的一次受控物理测试，显示这条垂直接近路径先撞到抽屉，无法到达下方杯子。不是历史规划器的拒绝日志，也不代表所有可能的顶抓路径都不可行。成功杯子视频及失败视频均为仿真背景。
- 微波炉：保留两条已验证的不同高度接触成功录像。

## 附件

- `images/`：杯子成功抓取的 1080p 图片及顶抓受阻的 720p 图片。
- `metadata/manifest.json`：视频清单、来源、分辨率和验证结果。
- `metadata/drawer_runtime_bindings.json`：从仿真实例导出的柜子位姿、上层关节、把手坐标系和成功条件。
- `metadata/drawer_task_contract.json`：根据用户任务意图人工编写的结构化任务定义。**不是人类视频模型自动提取的知识输出**，也不是旧版 bridge 编译器的 CLOSE 输出。
- `trajectories/`：新增仿真轨迹及接触记录。水果回位使用限速关节位置控制，目标保存在 `joint_position_command`；仅重放 7D 笛卡尔 action 不足以重现这部分。

这是视频素材包，不是包含全部 USD 资产的独立仿真环境。元数据里的原始项目绝对路径仅用于追溯来源。
'''
    (dst/'README.md').write_text(readme)
    print(json.dumps(dict(destination=str(dst),videos=len(entries),validated=True),indent=2))


if __name__=='__main__':main()
