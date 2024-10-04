import streamlit as st
import instructor
from anthropic import Anthropic
from pydantic import BaseModel
import os
import time

# Set up the Anthropic client with instructor
client = instructor.from_anthropic(Anthropic(api_key=""), mode=instructor.mode.Mode.ANTHROPIC_JSON)
class StepResponse(BaseModel):
    title: str
    content: str
    next_action: str
    confidence: float

def make_api_call(system_prompt, messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=max_tokens,
                temperature=0.2,
                system=system_prompt,
                messages=messages,
                response_model=StepResponse
            )
            return response
        except Exception as e:
            if attempt == 2:
                return StepResponse(
                    title="Error",
                    content=f"Failed to generate {'final answer' if is_final_answer else 'step'} after 3 attempts. Error: {str(e)}",
                    next_action="final_answer",
                    confidence=0.5
                )
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt):
    system_prompt = """You are an AI assistant that explains your reasoning step by step, incorporating dynamic Chain of Thought (CoT), reflection, and verbal reinforcement learning. Follow these instructions:

1. Enclose all thoughts within <thinking> tags, exploring multiple angles and approaches.
2. Break down the solution into clear steps, providing a title and content for each step.
3. After each step, decide if you need another step or if you're ready to give the final answer.
4. Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress.
5. Regularly evaluate your progress, being critical and honest about your reasoning process.
6. Assign a quality score between 0.0 and 1.0 to guide your approach:
   - 0.8+: Continue current approach
   - 0.5-0.7: Consider minor adjustments
   - Below 0.5: Seriously consider backtracking and trying a different approach
7. If unsure or if your score is low, backtrack and try a different approach, explaining your decision.
8. For mathematical problems, show all work explicitly using LaTeX for formal notation and provide detailed proofs.
9. Explore multiple solutions individually if possible, comparing approaches in your reflections.
10. Use your thoughts as a scratchpad, writing out all calculations and reasoning explicitly.
11. Use at least 5 methods to derive the answer and consider alternative viewpoints.
12. Be aware of your limitations as an AI and what you can and cannot do.

After every 3 steps, perform a detailed self-reflection on your reasoning so far, considering potential biases and alternative viewpoints."""

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(system_prompt, messages, 750)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        steps.append((f"Step {step_count}: {step_data.title}", 
                      step_data.content, 
                      thinking_time, 
                      step_data.confidence))
        
        # Add the assistant's response to the messages
        messages.append({"role": "assistant", "content": step_data.model_dump_json()})
        
        if step_data.next_action == 'final_answer' and step_count < 15:
            messages.append({"role": "user", "content": "Please continue your analysis with at least 5 more steps before providing the final answer."})
        elif step_data.next_action == 'final_answer':
            break
        elif step_data.next_action == 'reflect' or step_count % 3 == 0:
            messages.append({"role": "user", "content": "Please perform a detailed self-reflection on your reasoning so far, considering potential biases and alternative viewpoints."})
        else:
            messages.append({"role": "user", "content": "Please continue with the next step in your analysis."})
        
        step_count += 1

        yield steps, None

    messages.append({"role": "user", "content": "Please provide a comprehensive final answer based on your reasoning above, summarizing key points and addressing any uncertainties."})
    
    start_time = time.time()
    final_data = make_api_call(system_prompt, messages, 750, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data.content, thinking_time, final_data.confidence))

    yield steps, total_thinking_time

def main():
    st.set_page_config(page_title="Claude Reasoning Chain", page_icon="🧠", layout="wide")
    
    st.title("Claude Reasoning Chain: Extended self-reflection and analysis")
    
    st.markdown("""
    This is an improved prototype using prompting to create reasoning chains with extended self-reflection to improve output accuracy. It now thinks for longer periods and provides more detailed analysis, powered by Anthropic's Claude API.
    """)
    
    # Text input for user query
    user_query = st.text_input("Enter your query:", placeholder="e.g., What are the potential long-term effects of climate change on global agriculture?")
    
    if user_query:
        st.write("Generating response... This may take a while due to extended thinking time.")
        
        # Create empty elements to hold the generated text and total time
        response_container = st.empty()
        time_container = st.empty()
        
        # Generate and display the response
        for steps, total_thinking_time in generate_response(user_query):
            with response_container.container():
                for i, (title, content, thinking_time, confidence) in enumerate(steps):
                    if title.startswith("Final Answer"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                        st.markdown(f"**Confidence:** {confidence:.2f}")
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                            st.markdown(f"**Confidence:** {confidence:.2f}")
                            st.markdown(f"**Thinking time:** {thinking_time:.2f} seconds")
            
            # Only show total time when it's available at the end
            if total_thinking_time is not None:
                time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")

if __name__ == "__main__":
    main()