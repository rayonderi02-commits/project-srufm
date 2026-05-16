# Hybrid Accent Dataset Plan

The accent classifier uses three labels:

| Target label | Data source |
| --- | --- |
| `coastal` | Common Voice Swahili `sw-kimvita` plus local USB microphone recordings |
| `upcountry` | Common Voice Swahili `sw-barake` plus local USB microphone recordings |
| `nairobi` | Local USB microphone recordings; KenSpeech only if a subset is manually labelled as Nairobi |

Do not label all KenSpeech clips as Nairobi unless speaker origin/accent is known.
Otherwise the model may learn dataset differences instead of accent differences.

## Common Voice Setup

Download and extract Common Voice Swahili under:

```bash
/home/kiswahili-pi/project-srufm/speech_recognition_project/data/raw/
```

Then build hybrid metadata from the speech project directory:

```bash
cd /home/kiswahili-pi/project-srufm/speech_recognition_project
python3 scripts/build_hybrid_accent_metadata.py \
  --common-voice-root data/raw/cv-corpus-25.0-2026-03-09/sw \
  --local-metadata data/accent_metadata.csv \
  --data-dir data/raw \
  --output data/hybrid_accent_metadata.csv
```

If your extracted Common Voice folder has a different path, pass that folder as
`--common-voice-root`. It should contain files such as `validated.tsv` and a
`clips/` directory.

## Training

Train the accent model using the hybrid metadata:

```bash
python3 main.py train \
  --target accent \
  --model svm \
  --data-dir data/raw \
  --metadata data/hybrid_accent_metadata.csv \
  --save-dir models
```

The hardware runner will use:

- `models/accent_svm_model.joblib`
- `models/accent_scaler.joblib`
- `models/accent_label_encoder.joblib`
