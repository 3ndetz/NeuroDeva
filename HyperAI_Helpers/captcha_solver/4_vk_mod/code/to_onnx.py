"""
Convert model to ONNX format (from tf)
IMPORTANT!!!!
python 3.10 is not supported yet ( UPDATE: now it is ).
if you'll have a lot of undefined errors like me, then try using this versions of libraries:
python3.9
tensorflow==2.9.1
tf2onnx==1.11.1
onnxruntime==1.9.0
onnxconverter-common==1.10.0
_____________________Installing:____________________
pip install git+https://github.com/microsoft/onnxconverter-common
pip install onnxruntime
pip install pip install -U tf2onnx  or  pip install git+https://github.com/onnx/tensorflow-onnx
python -m tf2onnx.convert --saved-model "PATH_TO_MODEL" --output "PATH_TO_MODEL/model.onnx"

"""


def convertToOnnx():
    import pathlib, tf2onnx.convert, sys
    # from tools import MODEL_FNAME
    from utils.config import MODEL_FNAME, OUTPUT_ONNX

    dir = pathlib.Path(__file__).parent
    sys.argv.append("--saved-model")
    sys.argv.append(str(dir / MODEL_FNAME))
    sys.argv.append("--output")
    sys.argv.append(str(dir / OUTPUT_ONNX))

    tf2onnx.convert.main()


if __name__ == "__main__":
    convertToOnnx()
