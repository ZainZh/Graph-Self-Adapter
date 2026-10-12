#!/usr/bin/env python3
"""Explicit operator launcher for the separately bound engineering frame layer."""
import importlib.util
from pathlib import Path
import sys


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv);first_only='--first-stage-only' in args
    if first_only:args.remove('--first-stage-only')
    here=Path(__file__).resolve().parent
    operator=module(here/'operator_launcher.py','_object_centered_operator')
    original_load=operator.load_cli
    def load(package):
        cli=original_load(package);runtime=module(here/'object_centered_runtime.py','_object_centered_runtime')
        class ScopedExecutor(runtime.ObjectCenteredExecutor):first_stage_only=first_only
        cli.Executor=ScopedExecutor;prepare=cli.prepare_freeze
        def bound_prepare(*a,**kw):
            freeze=prepare(*a,**kw)
            for source in (Path(__file__),here/'object_centered_runtime.py'):
                freeze['bindings'].append({'path':str(source.resolve()),'sha256':cli.digest(source)})
            freeze['engineering_reference_frame_contract']=runtime.VARIANT;freeze['first_stage_only']=first_only
            return freeze
        cli.prepare_freeze=bound_prepare
        return cli
    operator.load_cli=load
    return operator.main(args)

if __name__=='__main__':main()
