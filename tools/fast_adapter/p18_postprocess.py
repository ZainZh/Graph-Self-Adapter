"""Run the prewritten reports exactly once after the complete frozen cohort."""
import json,subprocess,time
from pathlib import Path

root=Path('evaluations/fast_adapter_p18_visual_v01')
while not (root/'FINAL_RECEIPT.json').exists():
 if (root/'RESUME_ERROR.json').exists():raise RuntimeError((root/'RESUME_ERROR.json').read_text())
 time.sleep(15)
assert json.loads((root/'FINAL_RECEIPT.json').read_text())['status']=='completed_all_15_final_and_15_dev'
subprocess.run(['/home/zheng/miniforge3/envs/RoboTwin/bin/python3','-m','tools.fast_adapter.p18_report'],check=True)
subprocess.run(['python3','-m','tools.fast_adapter.p18_audit'],check=True)
print('REPORT_AND_INDEPENDENT_AUDIT_COMPLETE',flush=True)
