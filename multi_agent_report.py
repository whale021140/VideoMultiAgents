import os
import time
import json
import operator
import functools
from typing import Annotated, Sequence, TypedDict, List, Any
from langgraph.graph import StateGraph
from langchain.agents import AgentExecutor
# [REAL API - GEMINI] Replaced create_openai_tools_agent with Gemini version
from langchain_gemini_agent import create_gemini_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
# [REAL API - GEMINI] Replaced ChatOpenAI with ChatGemini
from langchain_gemini_wrapper import ChatGemini
# [REAL API - GEMINI] Import ask_gemini_omni for multi-modal analysis
from util import ask_gemini_omni, post_process, create_question_sentence

from tools.retrieve_video_clip_captions import retrieve_video_clip_captions
from tools.analyze_video_gpt4o import analyze_video_gpt4o
from tools.analyze_video_gemini import analyze_video_gemini
from tools.retrieve_video_scene_graph import retrieve_video_scene_graph
from tools.dummy_tool import dummy_tool
from util import post_process, ask_gpt4_omni, prepare_intermediate_steps, create_question_sentence

from dotenv import load_dotenv
load_dotenv()

# [REAL API - GEMINI] Use GEMINI_API_KEY instead of OPENAI_API_KEY
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("[REAL API - GEMINI] GEMINI_API_KEY environment variable not set")

# [REAL API - GEMINI] Initialize ChatGemini instead of ChatOpenAI
llm = ChatGemini(
    api_key=gemini_api_key,
    model_name='gemini-2.0-flash',
    temperature=0.0
)

llm_openai = ChatGemini(
    api_key=gemini_api_key,
    model_name='gemini-2.0-flash',
    temperature=0.7
)

def create_agent(llm, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    # [REAL API - GEMINI] Replaced create_openai_tools_agent with create_gemini_tools_agent
    agent = create_gemini_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True) # to return intermediate steps
    return executor

def agent_node(state, agent, name):
    print("****************************************")
    print(f"Executing {name} node!")
    print (f"State: {state}")
    print("****************************************")
    result = agent.invoke(state)

    # Extract tool results
    intermediate_steps = prepare_intermediate_steps(result.get("intermediate_steps", []))

    # Combine output and intermediate steps
    combined_output = f"Output:\n{result['output']}\n\nIntermediate Steps:\n{intermediate_steps}"

    return {"messages": [HumanMessage(content=combined_output, name=name)]}

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def mas_result_to_dict(result_data):
    log_dict = {}
    for message in result_data["messages"]:
        log_dict[message.name] = message.content
    return log_dict


def load_json_file(file_path):
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def execute_multi_agent(use_summary_info):
    # Load target question
    target_question_data = json.loads(os.getenv("QA_JSON_STR"))
    if os.getenv("DATASET") == "nextqa":
        video_id = target_question_data["q_uid"]
    elif os.getenv("DATASET") == "egoschema":
        video_id = os.getenv("VIDEO_INDEX")
    elif os.getenv("DATASET") == "demo":  # [DEMO MODE] Handle demo dataset
        video_id = os.getenv("VIDEO_INDEX", "demo_video_001")
    elif os.getenv("DATASET") == "real_mode":  # [REAL MODE - GEMINI] Real workflow demo
        video_id = target_question_data.get("q_uid", target_question_data.get("video_id"))

    # Load precomputed single agent results
    base_path = "data/results/"
    if os.getenv("DATASET") == "nextqa":
        video_file = os.path.join(base_path, "nextqa_val_single_video.json")
        text_file = os.path.join(base_path, "nextqa_val_single_text.json")
        graph_file = os.path.join(base_path, "nextqa_val_single_graph.json")
    elif os.getenv("DATASET") == "egoschema":
        video_file = os.path.join(base_path, "egoschema_fullset_single_video.json")
        text_file = os.path.join(base_path, "egoschema_fullset_single_text.json")
        graph_file = os.path.join(base_path, "egoschema_fullset_single_graph.json")
    elif os.getenv("DATASET") == "demo":  # [DEMO MODE] Use mock data paths
        video_file = "demo_data/results/demo_results.json"
        text_file = "demo_data/results/demo_results.json"
        graph_file = "demo_data/results/demo_results.json"
    elif os.getenv("DATASET") == "real_mode":  # [REAL MODE - GEMINI] Real workflow with 3 modalities
        video_file = os.path.join(base_path, "real_mode_single_video.json")
        text_file = os.path.join(base_path, "real_mode_single_text.json")
        graph_file = os.path.join(base_path, "real_mode_single_graph.json")
    else:
        # Default fallback
        video_file = os.path.join(base_path, f"{os.getenv('DATASET', 'unknown')}_single_video.json")
        text_file = os.path.join(base_path, f"{os.getenv('DATASET', 'unknown')}_single_text.json")
        graph_file = os.path.join(base_path, f"{os.getenv('DATASET', 'unknown')}_single_graph.json")
    
    video_data = load_json_file(video_file)
    text_data = load_json_file(text_file)
    graph_data = load_json_file(graph_file)
    
    if not all([video_data, text_data, graph_data]):
        print("Error: Failed to load one or more data files.")
        return -1, {}, {}

    # Initialize agents_result_dict
    agents_result_dict = {}
    
    # Check if the video_id exists in all three datasets
    if video_id in video_data and video_id in text_data and video_id in graph_data:
        print(f'{video_id} exists in all three datasets')
        # Get predictions from each modality
        video_pred = video_data[video_id].get("pred", -1)
        text_pred = text_data[video_id].get("pred", -1)
        graph_pred = graph_data[video_id].get("pred", -1)

        print(f"video_pred: {video_pred}, text_pred: {text_pred}, graph_pred: {graph_pred}")
        
        agents_result_dict = {
            "agent1": video_data[video_id]["response"].get("output", f"Prediction: Option {['A', 'B', 'C', 'D', 'E'][video_pred]}") + f"\n\n{json.dumps(video_data[video_id]['response'].get('intermediate_steps', ''), indent=2)}",
            "agent2": text_data[video_id]["response"].get("output", f"Prediction: Option {['A', 'B', 'C', 'D', 'E'][text_pred]}"),
            "agent3": graph_data[video_id]["response"].get("output", f"Prediction: Option {['A', 'B', 'C', 'D', 'E'][graph_pred]}"),
            "organizer": f"Predictions: Agent1={['A', 'B', 'C', 'D', 'E'][video_pred]}, Agent2={['A', 'B', 'C', 'D', 'E'][text_pred]}, Agent3={['A', 'B', 'C', 'D', 'E'][graph_pred]}"
        }

        # Check if all predictions are valid
        if all(pred != -1 for pred in [video_pred, text_pred, graph_pred]):
            # Check if all agents agree
            if video_pred == text_pred == graph_pred:
                print("✅ All agents agree! Directly returning the agreed answer.")
                prediction_result = video_pred
                
                # Create empty agent prompts dictionary
                agent_prompts = {
                    "agent1_prompt": "",
                    "agent2_prompt": "",
                    "agent3_prompt": "",
                    "organizer_prompt": ""
                }
                
                print(f"Truth: {target_question_data['truth']}, Pred: {prediction_result} (Option {['A', 'B', 'C', 'D', 'E'][prediction_result]})")
                return prediction_result, agents_result_dict, agent_prompts
            else:
                # Agents disagree - need to use Gemini Organizer
                print(f"⚠️  Agents disagree! Using Gemini Organizer to reconcile predictions...")
                print(f"   Agent1 (Video): {['A', 'B', 'C', 'D', 'E'][video_pred]}")
                print(f"   Agent2 (Text):  {['A', 'B', 'C', 'D', 'E'][text_pred]}")
                print(f"   Agent3 (Graph): {['A', 'B', 'C', 'D', 'E'][graph_pred]}")

    # Use Gemini Organizer to analyze agent results and determine final answer
    # [REAL API - GEMINI] Build prompt for Gemini Organizer
    agent_discussions = ""
    for agent in agents_result_dict:
        if agent != "organizer":
            agent_discussions += f"{agent}: {agents_result_dict[agent]}\n\n"

    gemini_prompt = f"""
You are an intelligent decision-making organizer coordinating three specialist agents analyzing a video question answering task.

Question: {create_question_sentence(target_question_data, shuffle_questions=False)}

Agent Predictions:
{agent_discussions}

Your task:
1. Analyze each agent's reasoning and prediction
2. Identify areas of agreement and disagreement
3. Synthesize the evidence to determine the most likely correct answer
4. Provide your final decision as one of [Option A, Option B, Option C, Option D, Option E]

Reasoning Process (step by step):
- What evidence supports each agent's position?
- Are there any inconsistencies or errors in the agents' reasoning?
- What is the most reliable prediction based on the evidence?

Final Answer: Please provide your decision in the format "FINAL ANSWER: [Option X]"
"""
    
    try:
        print("\n" + "="*80)
        print("[REAL API - GEMINI] Organizer Prompt:")
        print("="*80)
        print(gemini_prompt)
        print("="*80 + "\n")
        
        # [REAL API - GEMINI] Call ask_gemini_omni for multi-modal analysis
        gemini_result = ask_gemini_omni(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            prompt_text=gemini_prompt,
            temperature=0.0
        )
        
        print("[REAL API - GEMINI] Organizer Result:")
        print("="*80)
        print(gemini_result)
        print("="*80 + "\n")
        
        agents_result_dict["organizer"] = gemini_result
        
        # Extract the answer from Gemini's response
        if "FINAL ANSWER:" in gemini_result:
            answer_part = gemini_result.split("FINAL ANSWER:")[-1].strip()
            prediction_result = post_process(answer_part)
        else:
            # Fallback: try to extract Option from the response
            prediction_result = post_process(gemini_result)
            
    except Exception as e:
        print(f"❌ Error using Gemini Organizer: {e}")
        print("Falling back to first agent's prediction...")
        if video_id in video_data:
            prediction_result = video_data[video_id].get("pred", 0)
        else:
            prediction_result = 0
    
    print("="*80)
    if os.getenv("DATASET") in ["egoschema", "nextqa", "real_mode"]:
        if 0 <= prediction_result <= 4:
            print(f"✅ Final Decision: Truth={target_question_data['truth']}, Pred={prediction_result} (Option {['A', 'B', 'C', 'D', 'E'][prediction_result]})")
        else:
            print("⚠️  Error: Invalid prediction result value")
    elif os.getenv("DATASET") == "momaqa":
        print(f"✅ Final Decision: Truth={target_question_data['truth']}, Pred={prediction_result}")
    elif os.getenv("DATASET") == "demo":
        if 0 <= prediction_result <= 4:
            print(f"✅ Demo Result: Pred={prediction_result} (Option {['A', 'B', 'C', 'D', 'E'][prediction_result]})")
    print("="*80 + "\n")

    return prediction_result, agents_result_dict, {
        "agent1_prompt": "",
        "agent2_prompt": "",
        "agent3_prompt": "",
        "organizer_prompt": gemini_prompt
    }

if __name__ == "__main__":

    execute_multi_agent()
