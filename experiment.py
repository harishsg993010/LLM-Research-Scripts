import streamlit as st
import groq
import os
import json
import time

os.environ["GROQ_API_KEY"] = ""

client = groq.Groq()

def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt):
    # System prompt incorporating dynamic Chain of Thought (CoT), reflection, and verbal reinforcement learning
    messages = [
        {"role": "system", "content": """You are an AI assistant that explains your reasoning step by step, incorporating dynamic Chain of Thought (CoT), reflection, and verbal reinforcement learning. Follow these instructions:

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

After every 3 steps, perform a detailed self-reflection on your reasoning so far, considering potential biases and alternative viewpoints.

Respond in JSON format with 'title', 'content', 'next_action' (either 'continue', 'reflect', or 'final_answer'), and 'confidence' (a number between 0 and 1) keys.

Example of a valid JSON response:
```json
{
    "title": "Identifying Key Information",
    "content": "To begin solving this problem, we need to carefully examine the given information and identify the crucial elements that will guide our solution process. This involves...",
    "next_action": "continue",
    "confidence": 0.8
}```

Your goal is to demonstrate a thorough, adaptive, and self-reflective problem-solving process, emphasizing dynamic thinking and learning from your own reasoning.
"""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 750)  # Increased max_tokens for each step
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        # Handle the case where 'confidence' key is not present
        confidence = step_data.get('confidence', 0.5)  # Default to 0.5 if not present
        
        steps.append((f"Step {step_count}: {step_data.get('title', 'Untitled Step')}", 
                      step_data.get('content', 'No content provided'), 
                      thinking_time, 
                      confidence))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        next_action = step_data.get('next_action', 'continue')
        
        if next_action == 'final_answer' and step_count < 15:  # Increased minimum steps to 15
            messages.append({"role": "user", "content": "Please continue your analysis with at least 5 more steps before providing the final answer."})
        elif next_action == 'final_answer':
            break
        elif next_action == 'reflect' or step_count % 3 == 0:
            messages.append({"role": "user", "content": "Please perform a detailed self-reflection on your reasoning so far, considering potential biases and alternative viewpoints."})
        
        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide a comprehensive final answer based on your reasoning above, summarizing key points and addressing any uncertainties."})
    
    start_time = time.time()
    final_data = make_api_call(messages, 750, is_final_answer=True)  # Increased max_tokens for final answer
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    # Handle the case where 'confidence' key is not present in final_data
    final_confidence = final_data.get('confidence', 1.0)
    
    steps.append(("Final Answer", final_data.get('content', 'No final answer provided'), thinking_time, final_confidence))

    yield steps, total_thinking_time

def main():
    st.set_page_config(page_title="g1 prototype", page_icon="🧠", layout="wide")
    
    st.title("g1: Using Llama-3.1 70b on Groq to create o1-like reasoning chains with extended self-reflection")
    
    st.markdown("""
    This is an improved prototype of using prompting to create o1-like reasoning chains with extended self-reflection to improve output accuracy. It now thinks for longer periods and provides more detailed analysis. It is powered by Groq for fast reasoning steps!
                
    Open source [repository here](https://github.com/bklieger-groq)
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