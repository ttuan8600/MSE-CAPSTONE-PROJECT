"""Browser demonstration: replay held-out subjects, projected.

``scripts/demo_replay.py`` does the same job in a terminal. This exists because a
terminal on a projector is legible to the presenter and not to row six. Same
models, same held-out subjects, same code path -- only the rendering differs.

Deliberately separate from ``app.py``. That module serves the model of record
over JSON and is covered by the test suite; a presentation aid should not be able
to break it.

    python scripts/demo_server.py            # then open http://localhost:5001

Press SPACE or click for the next trial. The ground truth is hidden until the
prediction is on screen, so the audience sees the model commit before it is
marked -- which is the difference between a demonstration and an assertion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template_string

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.demo_replay import (
    AUDIO_CHECKPOINT,
    EEG_CHECKPOINT,
    HELD_OUT_SUBJECTS,
    build_dataset,
    load_predictors,
)
from src.preprocessing.eav_labels import EMOTION_NAMES

#: Figures of record, shown in the footer so the room never mistakes a 20-trial
#: sample for the result. See docs/CHANGELOG.md.
CV_FUSION = "67.02% +/- 1.11"
CV_AUDIO = "62.77% +/- 1.52"
CV_DELTA = "+4.25pp +/- 0.41 (p = 0.0031)"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>EmoAI - held-out subject replay</title>
<style>
  :root{
    --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --surface:#fff;
    --grid:#e1e0d9; --blue:#2a78d6; --orange:#eb6834;
    --good:#0f7a54; --bad:#c0392b;
  }
  *{box-sizing:border-box}
  body{margin:0;padding:2vh 3vw;font:16px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
       background:var(--surface);color:var(--ink)}
  header{display:flex;justify-content:space-between;align-items:baseline;
         border-bottom:2px solid var(--grid);padding-bottom:.6rem}
  h1{font-size:1.5rem;margin:0;letter-spacing:-.01em}
  .meta{color:var(--muted);font-size:.95rem}
  .trial{display:flex;gap:1.5rem;align-items:baseline;margin:1.4rem 0 .4rem}
  .trial b{font-size:1.15rem}
  .truth{font-size:1.6rem;font-weight:700}
  .hidden-truth{color:var(--muted);font-weight:400;font-size:1.1rem}
  .panels{display:grid;grid-template-columns:1.55fr 1fr;gap:2rem;margin-top:.8rem}
  .panel h2{font-size:1rem;margin:0 0 .7rem;color:var(--ink2);
            text-transform:uppercase;letter-spacing:.06em}
  .row{display:grid;grid-template-columns:7.5rem 3.6rem 1fr;align-items:center;
       gap:.6rem;margin:.42rem 0}
  .name{font-size:1.05rem}
  .pct{font-variant-numeric:tabular-nums;text-align:right;font-size:1.05rem}
  .bar{height:1.35rem;background:var(--grid);position:relative;border-radius:3px}
  .bar span{display:block;height:100%;background:var(--blue);border-radius:3px;
            transition:width .35s ease}
  .row.is-true .name{font-weight:700}
  .row.is-true .bar span{background:var(--good)}
  .row.is-wrong .bar span{background:var(--bad)}
  .verdict{font-size:1.5rem;font-weight:700;margin-top:1rem}
  .verdict.ok{color:var(--good)} .verdict.no{color:var(--bad)}
  .call{font-size:1.9rem;font-weight:700;margin:.2rem 0}
  .sub{color:var(--ink2);font-size:1.05rem}
  .tally{display:flex;gap:2.5rem;margin-top:1.6rem;padding-top:1rem;
         border-top:2px solid var(--grid);font-variant-numeric:tabular-nums}
  .tally div span{display:block;color:var(--muted);font-size:.85rem;
                  text-transform:uppercase;letter-spacing:.06em}
  .tally div b{font-size:1.7rem}
  footer{margin-top:1.6rem;color:var(--muted);font-size:.9rem;line-height:1.6}
  kbd{background:var(--grid);border-radius:3px;padding:.1rem .4rem;font-size:.85rem}
</style></head><body>
<header>
  <h1>Emotion recognition &mdash; subjects the model has never seen</h1>
  <div class="meta">held out: {{ subjects }} &nbsp;|&nbsp; chance 20.0%</div>
</header>

<div class="trial">
  <b id="counter">Press SPACE to begin</b>
  <span class="meta" id="provenance"></span>
</div>
<div class="truth" id="truth"><span class="hidden-truth">ground truth hidden until the model commits</span></div>

<div class="panels">
  <div class="panel">
    <h2>Late fusion &mdash; EEG + audio, 0 fusion parameters</h2>
    <div id="bars"></div>
    <div class="verdict" id="verdict"></div>
  </div>
  <div class="panel">
    <h2>Audio only</h2>
    <div class="call" id="audioCall">&mdash;</div>
    <div class="sub" id="audioSub"></div>
  </div>
</div>

<div class="tally">
  <div><span>Late fusion</span><b id="tf">0/0</b></div>
  <div><span>Audio only</span><b id="ta">0/0</b></div>
  <div><span>Models agree</span><b id="tg">0/0</b></div>
</div>

<footer>
  Figures of record, 7-fold subject-wise cross-validation over all 42 subjects,
  repeated across 3 seeds: late fusion <b>{{ cv_fusion }}</b>,
  audio only <b>{{ cv_audio }}</b>, paired difference <b>{{ cv_delta }}</b>.
  A handful of trials on screen is an illustration, not a measurement.<br>
  <kbd>SPACE</kbd> next trial &nbsp; <kbd>R</kbd> reset tally
</footer>

<script>
const EMOTIONS = {{ emotions|tojson }};
let n = 0, f = 0, a = 0, g = 0;
// Two-step: the first press shows what the models predicted, the second reveals
// the label and scores it. The audience watches the model commit before it is
// marked, which is what separates a demonstration from an assertion.
let stage = 'predict', current = null;

function showPrediction(d) {
  current = d;
  document.getElementById('counter').textContent = 'Trial ' + (n + 1);
  document.getElementById('provenance').textContent =
    'subject ' + d.subject_id + ', trial ' + d.trial_index +
    ' \\u00b7 EEG ' + d.eeg_shape + ' \\u00b7 audio ' + d.audio_shape;
  document.getElementById('truth').innerHTML =
    '<span class="hidden-truth">ground truth hidden \\u2014 press SPACE to reveal</span>';

  document.getElementById('bars').innerHTML = EMOTIONS.map(e =>
    '<div class="row" data-e="' + e + '"><div class="name">' + e + '</div>' +
    '<div class="pct">' + (d.probabilities[e]*100).toFixed(1) + '%</div>' +
    '<div class="bar"><span style="width:' + (d.probabilities[e]*100).toFixed(1) + '%"></span></div></div>'
  ).join('');

  document.getElementById('verdict').textContent = 'predicts ' + d.fusion;
  document.getElementById('verdict').className = 'verdict';
  document.getElementById('audioCall').textContent = d.audio;
  document.getElementById('audioSub').innerHTML =
    (d.audio_confidence*100).toFixed(1) + '% confidence';
  document.getElementById('audioSub').style.color = 'var(--ink2)';
  stage = 'reveal';
}

function revealTruth() {
  const d = current;
  n += 1; f += d.fusion_correct; a += d.audio_correct; g += d.agree;

  document.querySelectorAll('.row').forEach(row => {
    const e = row.dataset.e;
    if (e === d.truth) row.classList.add('is-true');
    else if (e === d.fusion) row.classList.add('is-wrong');
  });

  document.getElementById('truth').innerHTML = 'Ground truth: ' + d.truth;
  const v = document.getElementById('verdict');
  v.textContent = d.fusion_correct ? '\\u2713 CORRECT \\u2014 ' + d.fusion
                                   : '\\u2717 WRONG \\u2014 said ' + d.fusion;
  v.className = 'verdict ' + (d.fusion_correct ? 'ok' : 'no');
  document.getElementById('audioSub').innerHTML =
    (d.audio_confidence*100).toFixed(1) + '% confidence<br>' +
    (d.audio_correct ? '\\u2713 correct' : '\\u2717 wrong');
  document.getElementById('audioSub').style.color =
    d.audio_correct ? 'var(--good)' : 'var(--bad)';

  document.getElementById('tf').textContent = f + '/' + n;
  document.getElementById('ta').textContent = a + '/' + n;
  document.getElementById('tg').textContent = g + '/' + n;
  stage = 'predict';
}

function advance() {
  if (stage === 'reveal') revealTruth();
  else fetch('/trial').then(r => r.json()).then(showPrediction);
}
function reset() { n = f = a = g = 0; stage = 'predict';
  ['tf','ta','tg'].forEach(i => document.getElementById(i).textContent = '0/0'); }

document.addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); advance(); }
  if (e.key === 'r' || e.key === 'R') reset();
});
document.body.addEventListener('click', advance);
</script></body></html>"""


def create_demo_app(eeg_path: Path, audio_path: Path, seed: int) -> Flask:
    app = Flask(__name__)
    predictors = load_predictors(eeg_path, audio_path)
    dataset = build_dataset(list(HELD_OUT_SUBJECTS))
    rng = np.random.default_rng(seed)

    @app.route("/")
    def index():
        return render_template_string(
            PAGE,
            subjects=", ".join(str(s) for s in HELD_OUT_SUBJECTS),
            emotions=EMOTION_NAMES,
            cv_fusion=CV_FUSION,
            cv_audio=CV_AUDIO,
            cv_delta=CV_DELTA,
        )

    @app.route("/trial")
    def trial():
        sample = dataset[int(rng.integers(len(dataset)))]
        eeg = sample["eeg"].numpy()
        audio = sample["audio"].numpy()
        truth = EMOTION_NAMES[sample["emotion"]]

        fused = predictors["fusion"].predict(eeg_data=eeg, audio_data=audio)
        audio_only = predictors["audio"].predict(audio_data=audio)

        return jsonify(
            subject_id=int(sample["subject_id"]),
            trial_index=int(sample["trial_index"]),
            eeg_shape=str(tuple(eeg.shape)),
            audio_shape=str(tuple(audio.shape)),
            truth=truth,
            fusion=fused["emotion"],
            probabilities=fused["probabilities"],
            fusion_correct=fused["emotion"] == truth,
            audio=audio_only["emotion"],
            audio_confidence=audio_only["confidence"],
            audio_correct=audio_only["emotion"] == truth,
            agree=fused["emotion"] == audio_only["emotion"],
        )

    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eeg-checkpoint", type=Path, default=EEG_CHECKPOINT)
    parser.add_argument("--audio-checkpoint", type=Path, default=AUDIO_CHECKPOINT)
    args = parser.parse_args(argv)

    print("Loading models and held-out trials...")
    app = create_demo_app(args.eeg_checkpoint, args.audio_checkpoint, args.seed)
    print(f"\n  Demo ready:  http://localhost:{args.port}\n")
    # threaded=False keeps predictions serialised; the page is driven by one
    # keypress at a time and concurrency would only add a way to fail on stage.
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
