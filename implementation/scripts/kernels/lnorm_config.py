#
# Generate sw/LayerNorm/include/lnorm.h from a LayerNorm configuration file.
#
import os
import argparse
import importlib.util
import types
import utils.kernel_configuration as kc


def import_module_from_path(module_path):
    """
    Dynamically import a module from an absolute path and mimic `from module import *`.
    """
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot find a module at path: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))
    return module


parser = argparse.ArgumentParser(description="Generate C header files from a LayerNorm configuration file.")
parser.add_argument("input_file",  nargs="?", help="Path to LayerNorm configuration python file")
parser.add_argument("output_file", nargs="?", help="Path to LayerNorm configuration C header file")
args = parser.parse_args()

module = import_module_from_path(os.path.abspath(args.input_file))
print(f"Successfully imported: {module.__name__} from {os.path.abspath(args.input_file)}")

lnorm = LayerNorm()
if lnorm.dtype != 'fp16':
    raise ValueError("LayerNorm supports fp16 only")

# The generic generator casts every number to uint64_t, which would turn eps
# into 0; write eps as a float literal instead.
fields = {k: v for k, v in vars(lnorm).items() if k != 'eps'}
kc.generate_config_C_header("LNORM", types.SimpleNamespace(**fields), args.output_file, lnorm.dtype, 1,
                            appendix=[f'#define LNORM_EPS_F ({float(lnorm.eps)!r}f)'])
