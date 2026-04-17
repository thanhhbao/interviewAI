from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interview_ai.conversation import TurnBasedConversationManager
from interview_ai.pipeline import InterviewPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a turn-based interview session with push-to-talk style audio files.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--resume-file", required=True)
    parser.add_argument("--jd-file", required=True)
    parser.add_argument("--answer-audio", required=True, nargs="+")
    parser.add_argument("--session-output", required=True)
    parser.add_argument("--tts-output-dir", required=True)
    args = parser.parse_args()

    pipeline = InterviewPipeline(model_name=args.model_name, adapter_dir=args.adapter_dir)
    manager = TurnBasedConversationManager(llm_pipeline=pipeline)
    state = manager.start_session(args.resume_file, args.jd_file)

    for turn_index, audio_file in enumerate(args.answer_audio, start=1):
        tts_file = Path(args.tts_output_dir) / f"turn_{turn_index:02d}_question.txt"
        manager.speak_current_question(state, str(tts_file))
        state = manager.process_answer_turn(state, audio_file)
        print(json.dumps(manager.summarize_state(state), ensure_ascii=False, indent=2))
        if not state.current_question:
            break

    output_file = manager.save_state(state, args.session_output)
    print(f"Saved conversation state to {output_file}")


if __name__ == "__main__":
    main()
