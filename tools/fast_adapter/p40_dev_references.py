"""Finite development baselines, one task/GPU; output-driven tuning forbidden here."""
import argparse,subprocess,sys
from tools.fast_adapter.p40_launch_v03 import ROOT,environment
from tools.fast_adapter.p40_evaluate import STRATA

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    for split in STRATA:
        for arm in ('base','fixed0','fixed1','fixed2','fixed3'):
            with (ROOT/'logs'/f'dev_reference_{a.task}_{split}_{arm}.log').open('x') as f:
                subprocess.run([sys.executable,'-u','-m','tools.fast_adapter.p40_evaluate','--task',a.task,'--phase','dev','--split',split,'--kind','references','--seed','0','--arm',arm],env=environment(a.gpu),stdout=f,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':main()
