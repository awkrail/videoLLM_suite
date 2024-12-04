import argparse

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import videoLLM_suite

def main(input_path: str):
    model = videoLLM_suite.create_pipeline('video_llama_vicuna_7b')
    
    input_path = "birthday.mp4"
    text_prompt = "Describe this video for details."
    # end-to-end
    sentence = model.generate(input_path, text_prompt)
    
    # encode & generate text
    model.encode_video(input_path)
    sentence = model.generate_text(text_prompt)


    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str, required=True, help='input video path.')
    args = parser.parse_args()
    main(args.input_path)