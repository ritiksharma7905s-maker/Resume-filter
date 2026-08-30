from agents import Agent, Runner, function_tool, trace, AgentHooks
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from pypdf import PdfReader
import asyncio
import os
import weave
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
import gradio as gr

load_dotenv(override=True)


# Framework Infrastructure & Parsers
class ConsoleLogHooks(AgentHooks):
    async def on_start(self, context, agent):
        print(f"🚀 [Agent Started]: {agent.name} is thinking...")
    async def on_end(self, context, agent, result):
        print(f"✅ [Agent Completed]: {agent.name} finished processing.")

def read_pdf_bytes(file_path: str) -> str:
    reader = PdfReader(file_path)
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)

def parse_uploaded_file(file_obj) -> str:
    """Extracts text based on uploaded file format (gradio temp object)"""
    file_path = Path(file_obj.name)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_bytes(str(file_path))
    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8")
    return ""

# Agent Definitions
instructions1 = "You are a Senior HR in a reputed Organisation. Select the appropriate resume matching the job description."
instructions2 = "From the selected candidate details, return the candidate name and email in a markdown table with 'Name' and 'Email' columns."

hr_agent = Agent(name="HR Agent", instructions=instructions1, model="gpt-4o-mini", hooks=ConsoleLogHooks())
name_and_email_agent = Agent(name="Name and Email Agent", instructions=instructions2, model="gpt-4o-mini", hooks=ConsoleLogHooks())

subject_writer = Agent(name="Email subject writer", instructions="Write an email subject line.", model="gpt-4o-mini")
subject_tool = subject_writer.as_tool(tool_name="subject_writer", tool_description="Write an email subject line")

html_converter = Agent(name="HTML email body converter", instructions="Convert markdown email text to styled HTML.", model="gpt-4o-mini")
html_tool = html_converter.as_tool(tool_name="html_converter", tool_description="Convert a text email body to an HTML email body")

@function_tool
def send_html_email(subject: str, html_body: str, to_email: str) -> dict[str, str]:
    """ Send out an email with the given subject and HTML body to the chosen recipient candidate """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("dashyabhishek98@gmail.com")  
    to_recipient = To(to_email)  
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_recipient, subject, content).get()
    sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}

emailer_agent = Agent(
    name="Email Manager",
    instructions="Format the message as HTML, generate a subject, and send it to the candidate.",
    tools=[subject_tool, html_tool, send_html_email],
    model="gpt-4o-mini",
    hooks=ConsoleLogHooks()
)


# Web Interface
async def run_web_pipeline(jd_file, resume_files, run_outreach):
    # Initialize Weave tracking if API Key exists
    if os.environ.get('WANDB_API_KEY'):
        weave.init("abhishek98-psit-kanpur/intro-example")
        
    if not jd_file or not resume_files:
        return "Please upload the Job Description file and at least 1 Resume.", "Pipeline canceled."

    # Parse file streams
    job_description = parse_uploaded_file(jd_file)
    resumes_database = {}
    for f in resume_files:
        resumes_database[Path(f.name).name] = parse_uploaded_file(f)

    # Step 1: Run Selection agents
    combined_message = f"Job description:\n{job_description}\n\nResumes:\n{resumes_database}"
    hr_result = await Runner.run(hr_agent, combined_message)
    extraction_context = f"Resumes Database:\n{combined_message}\n\nHR Selection Details:\n{hr_result.final_output}"
    extraction_result = await Runner.run(name_and_email_agent, extraction_context)

    screening_summary = f"{hr_result.final_output}\n\n### Candidate Breakdown\n{extraction_result.final_output}"
    outreach_summary = "Outreach checkbox not selected. Email bypassed."

    # Step 2: Conditional Outreach
    if run_outreach:
        outreach_instruction = (
            f"From this candidate table data:\n{extraction_result.final_output}\n\n"
            f"Extract the 'Email' address value. Draft an interview invitation email "
            f"for a Technical Screening regarding the Job Description snippet: {job_description[:100]}... "
            f"and execute your tools to deliver it."
        )
        email_delivery_result = await Runner.run(emailer_agent, outreach_instruction)
        outreach_summary = email_delivery_result.final_output

    return screening_summary, outreach_summary

# ---------------------------------------------------------
# Gradio Layout Construction
# ---------------------------------------------------------
with gr.Blocks(title="AI Candidate Pipeline") as demo:
    gr.Markdown("# 🏢 Automated HR Screener & Outreach Agent")
    
    with gr.Row():
        with gr.Column(scale=1):
            jd_input = gr.File(label="Upload Job Description (.txt, .md, .pdf)", file_count="single")
            resumes_input = gr.File(label="Upload Candidate Resumes (.pdf)", file_count="multiple")
            outreach_toggle = gr.Checkbox(label="Automatically Email Selected Candidate via SendGrid", value=False)
            submit_btn = gr.Button("Process & Filter Resumes", variant="primary")
            
        with gr.Column(scale=2):
            screening_output = gr.Markdown(label="Screening Report Analysis")
            outreach_output = gr.Textbox(label="SendGrid Dispatch System Status", interactive=False)

    # Handle asynchronous execution pathway natively inside Gradio
    submit_btn.click(
        fn=run_web_pipeline,
        inputs=[jd_input, resumes_input, outreach_toggle],
        outputs=[screening_output, outreach_output]
    )

if __name__ == "__main__":
    demo.launch()