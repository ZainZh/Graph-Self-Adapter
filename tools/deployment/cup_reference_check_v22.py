#!/usr/bin/env python3
"""Workstation file-only V19 receipt check; target resources may be uninstalled."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from gemini_promotion_check import check,digest


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--config',type=Path);a=p.parse_args(argv)
    root=a.package.resolve();approved=(root/'local/ROOT_RECEIPT_SHA256.txt').read_text().strip();report=check(root,approved)
    cfg=json.loads((a.config or root/'local/CONFIG_REAL.json').read_text());resources={'sdk_file':cfg['sdk_file'],'left_calibration':cfg['left_calibration'],'right_calibration':cfg['right_calibration'],'sam_utility':cfg['vision']['sam_utility'],'sam_checkpoint':cfg['vision']['sam_checkpoint']};receipt=json.loads((root/'PROMOTION_RECEIPT.json').read_text())
    states={}
    for name,path in resources.items():
        file=Path(path);states[name]='not-installed-at-target-path-on-this-host' if not file.is_file() else 'hash-match' if digest(file)==receipt['external_resources'][name]['sha256'] else 'hash-mismatch'
    report.update(mode='check',task='put_mug_on_box',portable_internal_bindings_verified=True,target_external_resource_status=states,full_target_preflight_required_before_shadow_real=True,hardware_ready=False,device_connections=0,API_calls=0,key_read=False)
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':main()
