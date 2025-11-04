import os
import time
import argparse
from multiprocessing import Pool
from functools import partial
from util import save_result
from util import set_environment_variables
from util import read_json_file
from single_agent import execute_single_agent
import multi_agent_star
import multi_agent_report
import multi_agent_report_star
import multi_agent_debate
import traceback

# [DEMO MODE] - Check if running in demo mode with mock APIs
USE_MOCK_API = os.getenv('USE_MOCK_API', 'false').lower() == 'true'

# Import required tools for video analysis
# [DEMO MODE] - Import mock APIs if enabled
if USE_MOCK_API:
    print("[DEMO MODE] Using mock APIs for demonstration")
    from mock_apis.mock_openai import MockOpenAI
    from mock_apis.mock_gemini import MockGemini
    from mock_apis.mock_vision import MockVisionExtractor
else:
    from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
    from tools.analyze_video_gpt4o import analyze_video_gpt4o
    from tools.analyze_video_gemini import analyze_video_gemini
    from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
    from tools.analyze_all_gpt4o import analyze_all_gpt4o

def get_tools(modality):
    """
    Get the appropriate tools based on the modality.
    
    Args:
        modality: String indicating which tools to use ('video', 'text', 'graph', or 'all')
    
    Returns:
        List of tool functions to use for processing
    """
    if modality == "video":
        return [analyze_video_gemini]
    elif modality == "text":
        return [retrieve_video_clip_captions]
    elif modality == "graph":
        return [retrieve_video_scene_graph]
    elif modality == "all":
        return [analyze_all_gpt4o]
    else:
        raise ValueError(f"Unknown modality: {modality}")

def process_single_video(modality, agents, dataset, use_summary_info, video_data):
    """
    Process a single video with tools initialized inside the worker.
    
    Args:
        modality: String indicating which tools to use
        dataset: Name of the dataset being processed
        video_data: Tuple of (video_id, json_data)
    """
    video_id, json_data = video_data
    try:
        print(f"Processing video_id: {video_id}")
        print(f"JSON data: {json_data}")

        # Set environment variables for this process
        set_environment_variables(dataset, video_id, json_data)

        if agents == "single":
            # Initialize tools inside the worker process
            tools = get_tools(modality)
            # Execute video analysis
            result, agent_response, agent_prompts = execute_single_agent(tools, use_summary_info)
        elif agents.startswith("multi_report_star"):
            result, agent_response, agent_prompts = multi_agent_report_star.execute_multi_agent(use_summary_info)
        elif agents.startswith("multi_report"):
            result, agent_response, agent_prompts = multi_agent_report.execute_multi_agent(use_summary_info)
        elif agents.startswith("multi_star"):
            result, agent_response, agent_prompts = multi_agent_star.execute_multi_agent(use_summary_info)
        elif agents.startswith("multi_debate"):
            result, agent_response, agent_prompts = multi_agent_debate.execute_multi_agent_multi_round(use_summary_info)

        # Save results
        print(f"Results for video {video_id}: {result}")
        save_result(os.getenv("QUESTION_FILE_PATH"), video_id, agent_prompts, 
                   agent_response, result, save_backup=False)
        
        return True
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        print(traceback.format_exc())
        time.sleep(1)
        return False

def get_unprocessed_videos(question_file_path, max_items=1000):
    """
    Get a list of all unprocessed videos from the question file.
    
    Args:
        question_file_path: Path to the JSON file containing video questions
    
    Returns:
        List of tuples containing (video_id, json_data) for unprocessed videos
    """
    dict_data = read_json_file(question_file_path)
    unprocessed_videos = []
    for i, (video_id, json_data) in enumerate(list(dict_data.items())[:max_items]):
        # Skip metadata entries like "_comment"
        if not isinstance(json_data, dict):
            continue
        
        # Priority: check metadata.pred first (for real_mode), then root-level pred
        if "metadata" in json_data and isinstance(json_data["metadata"], dict):
            pred_value = json_data["metadata"].get("pred", None)
        else:
            pred_value = json_data.get("pred", None)
        
        # Include unprocessed videos (pred == -2)
        if pred_value == -2:
            unprocessed_videos.append((video_id, json_data))
    return unprocessed_videos

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Dataset to use for the analysis")
    parser.add_argument('--dataset', type=str, help="Example: egoschema, nextqa, etc.")
    parser.add_argument('--modality', type=str, help="Example: video, text, graph, all.")
    parser.add_argument('--agents', type=str, help="Example: single, multi_star.")
    parser.add_argument('--use_summary_info', type=bool, default=True, help="Use summary info.")
    parser.add_argument('--num_workers', type=int, default=1, 
                       help="Number of worker processes. Defaults to CPU count - 1")
    parser.add_argument('--max_items', type=int, default=999999999, 
                       help="Number of videos to process. Defaults to all.")
    args = parser.parse_args()

    # [DEMO MODE] - Handle demo dataset
    if args.dataset == "demo":
        print("\n" + "="*80)
        print("[DEMO MODE] Running in demonstration mode")
        print("="*80)
        print("Using mock APIs and demo data for testing")
        print("To use real models and datasets, set USE_MOCK_API=false and provide real API keys")
        print("="*80 + "\n")
        
        os.environ["QUESTION_FILE_PATH"] = "./demo_data/qa/demo_qa.json"
        os.environ["CAPTIONS_FILE"] = "./demo_data/captions/demo_captions.json"
        os.environ["GRAPH_DATA_PATH"] = "./demo_data/features/demo_features.json"
        os.environ["SUMMARY_CACHE_JSON_PATH"] = "./demo_data/features/demo_features.json"
        os.environ["VIDEO_DIR_PATH"] = "./demo_data/videos/"
        os.environ["FRAME_NUM"] = "8"
    # Set dataset-specific environment variables
    elif args.dataset == "egoschema":
        os.environ["QUESTION_FILE_PATH"] = f"path/to/egoschema/fullset_{args.agents}_{args.modality}.json"
        os.environ["CAPTIONS_FILE"] = "path/to/egoschema_captions_gpt4o_caption_guided.json"
        os.environ["GRAPH_DATA_PATH"] = "path/to/egoschema_graph_captions.json"
        os.environ["SUMMARY_CACHE_JSON_PATH"] = "path/to/egoschema_summary_cache.json"
        os.environ["VIDEO_DIR_PATH"] = "path/to/egoschema/videos"
        os.environ["FRAME_NUM"] = "180"
    elif args.dataset == "nextqa":
        os.environ["QUESTION_FILE_PATH"] = f"path/to/nextqa/val_{args.agents}_{args.modality}.json"
        os.environ["GRAPH_DATA_PATH"] = "path/to/nextqa_graph_captions_gpt4o.json"
        os.environ["CAPTIONS_FILE"] = "path/to/nextqa_captions_gpt4o_question_guided.json"
        os.environ["SUMMARY_CACHE_JSON_PATH"] = "path/to/nextqa_summary_cache_val.json"
        os.environ["IMAGES_DIR_PATH"] = "path/to/nextqa/frames_aligned/"
        os.environ["VIDEO_DIR_PATH"] = "path/to/NExTVideo"
        os.environ["FRAME_NUM"] = "180"
    elif args.dataset == "momaqa":
        os.environ["QUESTION_FILE_PATH"] = "path/to/momaqa_test_anno.json"
        os.environ["CAPTIONS_FILE"] = "path/to/momaqa_captions.json"
        os.environ["SUMMARY_CACHE_JSON_PATH"] = "path/to/momaqa_summary_cache.json"
        os.environ["GRAPH_DATA_PATH"] = "path/to/momaqa_graph_data.json"
        os.environ["IMAGES_DIR_PATH"] = "path/to/momaqa/images"
        os.environ["FRAME_NUM"] = "90"
    elif args.dataset == "intentqa":
        os.environ["QUESTION_FILE_PATH"] = f"path/to/intentqa_test_{args.agents}_{args.modality}.json"
        os.environ["GRAPH_DATA_PATH"] = "path/to/intentqa_graph_captions.json"
        os.environ["CAPTIONS_FILE"] = "path/to/intentqa_question_guided_captions.json"
        os.environ["SUMMARY_CACHE_JSON_PATH"] = "path/to/intentqa_summary_cache.json"
        os.environ["IMAGES_DIR_PATH"] = "path/to/images_nextqa"
        os.environ["FRAME_NUM"] = "180"
    elif args.dataset == "hourvideo":
        os.environ["QUESTION_FILE_PATH"] = "path/to/hourvideo_single_video.json"
        os.environ["CAPTIONS_FILE"] = "path/to/hourvideo_local_captions.json"
        os.environ["SUMMARY_CACHE_JSON_PATH"] = "path/to/hourvideo_summary_cache.json"
        os.environ["GRAPH_DATA_PATH"] = "path/to/hourvideo_graph_captions.json"
    elif args.dataset == "real_mode":  # [REAL MODE - GEMINI] Real workflow demo with true Gemini API
        print("\n" + "="*80)
        print("[REAL MODE - GEMINI] Running real workflow with Gemini API")
        print("="*80)
        print("Using true Gemini API with multi-modality agent disagreement resolution")
        print("Demonstrating Gemini's ability to reconcile conflicting predictions")
        print("="*80 + "\n")
        
        os.environ["QUESTION_FILE_PATH"] = "./real_mode_demo_qa.json"
        os.environ["CAPTIONS_FILE"] = "./demo_data/captions/demo_captions.json"
        os.environ["GRAPH_DATA_PATH"] = "./demo_data/features/demo_features.json"
        os.environ["SUMMARY_CACHE_JSON_PATH"] = "./demo_data/features/demo_features.json"
        os.environ["VIDEO_DIR_PATH"] = "./demo_data/videos/"
        os.environ["FRAME_NUM"] = "8"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # [DEMO MODE] - Set dataset environment variable (moved after condition for demo)
    os.environ["DATASET"] = args.dataset

    # Get list of unprocessed videos
    unprocessed_videos = get_unprocessed_videos(os.getenv("QUESTION_FILE_PATH"), max_items=args.max_items)
    
    # Determine number of worker processes
    num_workers = args.num_workers

    print(f"Starting processing with {num_workers} workers")
    
    # Create process pool and process videos in parallel
    with Pool(num_workers) as pool:
        # Create a partial function with fixed arguments
        process_func = partial(process_single_video, args.modality, args.agents, args.dataset, args.use_summary_info)
        
        # Process videos in parallel and collect results
        results = pool.map(process_func, unprocessed_videos)
        
        # Print summary
        successful = sum(1 for r in results if r)
        failed = len(results) - successful
        print(f"\nProcessing complete:")
        print(f"Successfully processed: {successful} videos")
        print(f"Failed to process: {failed} videos")

if __name__ == "__main__":
    main()
