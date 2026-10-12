#!/usr/bin/env python3
"""Prepared thin-real entry configuration check; never opens a device or service."""
import argparse
import json
from pathlib import Path
from macro_promotion_check import check


def prepare_real(package,receipt_sha,config_path):
    cfg=json.loads(Path(config_path).read_text());runtime=cfg.get('runtime',cfg)
    paths={'sam_checkpoint':runtime['vision']['sam_checkpoint'],'sam_utility':runtime['vision']['sam_utility'],
           'sdk_file':runtime['sdk_file'],'left_calibration':runtime['left_calibration'],'right_calibration':runtime['right_calibration']}
    result=check(package,receipt_sha,external_paths=paths);receipt=json.loads((Path(package)/'PROMOTION_RECEIPT.json').read_text())
    if runtime.get('robot_TCP_offset_m')!=.197:raise ValueError('real Tool1 robot TCP offset must be explicitly197mm')
    result.update(execution_variant=receipt['execution_variant'],graph_nodes=receipt['graph_nodes'],
        canonical_runtime_import='direct_white_runtime.DirectExecutor' if receipt['execution_variant']=='red_legacy_white_direct_box_v15' else 'handle_macro_runtime.MacroExecutor',
        source_pythonpath=str(Path(package).resolve()/'source'),data_root=str(Path(package).resolve()/'data'),portable_root=str(Path(package).resolve()/'package'),
        real_IO_plan=['existing ZMQ NativeRPC service; actor-only14rad/g commands','actual TCP30004 FramedFeedbackStream; gripper None remains unknown',
                      'ZED RGBD with live resized K +same actual SAM factory/checkpoint','same canonical executor, safety and current Root file bridge'],
        execution_activated=False,automatic_enable_home_tool_or_gripper_scan=False,
        operator_declarations_present={'safety_reviewed':runtime.get('hardware_safety_reviewed') is True,'initial_gripper_confirmed':runtime.get('initial_gripper_state_confirmed') is True},
        declarations_are_not_measured_contact_or_hardware_qualification=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--receipt-sha256',required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args(argv)
    print(json.dumps(prepare_real(a.package,a.receipt_sha256,a.config),indent=2))

if __name__=='__main__':main()
