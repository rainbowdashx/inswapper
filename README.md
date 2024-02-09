# inswapper

## Installation

```bash
# create a Python venv
python3 -m venv venv

# activate the venv
source venv/bin/activate

# install required packages
pip install -r requirements.txt
```

Install  ``onnxruntime-gpu`` for GPU inference

Install  ``onnxruntime`` for CPU inference

## API

```json
 {
    "input": {
        "source_img": "{{base64.replace('data:image/jpeg;base64,','')}}",
        "target_img": "{{base64.replace('data:image/jpeg;base64,','')}}",
    }
 }

```

With Face Restoration

```json
{
    "input": {
        "source_img": "{{base64.replace('data:image/jpeg;base64,','')}}",
        "target_img": "{{base64.replace('data:image/jpeg;base64,','')}}",
        "background_enhance": true,
        "face_upsample": true,
        "upscale": 2,
        "codeformer_fidelity": 0.5,
        "shouldRunRestoration": true
    }
}
```
