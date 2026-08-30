from agents import Agent, Runner, function_tool, trace, AgentHooks, add_trace_processor
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from pypdf import PdfReader
from agents.tracing.processors import ConsoleSpanExporter, BatchTraceProcessor
import asyncio
import os
import weave
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content


load_dotenv(override=True)
client = OpenAI()

# 2. APPLY OPTION 2 HERE: Initialize the background trace processor
console_exporter = ConsoleSpanExporter()
console_processor = BatchTraceProcessor(exporter=console_exporter)
add_trace_processor(console_processor)

# 2. Define a custom hook class to print out framework steps as they happen
class ConsoleLogHooks(AgentHooks):
    async def on_start(self, context, agent):
        print(f"🚀 [Agent Started]: {agent.name} is thinking...")
        
    async def on_end(self, context, agent, result):
        print(f"✅ [Agent Completed]: {agent.name} finished processing.")

instructions1 = "You are a Senior HR in a reputed Organisation. \
    you need to select the appropriate resume which matches the job description, \
    which is most relevant to the job description, \
    which is most likely to get the job."

instructions2 = "Now as we have selected the appropriate resume from hr_agent, you need to give the name of the candidate and the email id of the candidate in the form of table having the columns 'Name' and 'Email'."

hr_agent = Agent(
    name="HR Agent",
    instructions=instructions1,
    model="gpt-4o-mini",
    hooks=ConsoleLogHooks() # Attach log handler here!
)

name_and_email_agent = Agent(
    name="Name and Email Agent",
    instructions=instructions2,
    model="gpt-4o-mini",
    hooks=ConsoleLogHooks() # Attach log handler here!
)

# hr_agent_tool = hr_agent.as_tool(tool_name="hr_agent", tool_description=instructions1)
# name_and_email_agent_tool = name_and_email_agent.as_tool(tool_name="name_and_email_agent", tool_description=instructions2)

# tools = [hr_agent_tool, name_and_email_agent_tool]


async def select_resume(message: str):

    with trace("Selection Resume"):
        # FIX: Pass the raw Agent instances directly to Runner.run instead of the tool wrappers
        hr_result = await Runner.run(hr_agent, message)

        # Build sequential context so the extraction agent knows who was picked
        extraction_context = f"Resumes Database:\n{message}\n\nHR Selection Details:\n{hr_result.final_output}"
        extraction_result = await Runner.run(name_and_email_agent, extraction_context)
        
    return [hr_result, extraction_result]


def read_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def read_job_description_file(file_path: str | Path) -> str:
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Job description file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in {".txt", ".md"}:
        return read_text_file(path)

    raise ValueError(
        f"Unsupported job description format: {suffix}. Use .txt, .md, or .pdf"
    )


def find_pdf_files(folder: str | Path) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder_path}")
    return sorted(folder_path.glob("*.pdf"))


def read_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def read_pdfs_from_folder(folder: str | Path) -> dict[str, str]:
    resumes = {}
    for pdf_path in find_pdf_files(folder):
        print(f"Reading: {pdf_path.name}")
        try:
            resumes[pdf_path.name] = read_pdf(pdf_path)
            print(f"  -> {len(resumes[pdf_path.name])} characters extracted")
        except Exception as e:
            print(f"  -> Failed to read {pdf_path.name}: {e}")
    return resumes

def load_resumes(folder: str | Path) -> dict[str, str]:
    
    pdf_files = find_pdf_files(folder)
    if not pdf_files:   
        print(f"No PDF files found in: {folder}")
    else:
        print(f"\nFound {len(pdf_files)} PDF file(s) in {folder}\n")
        resumes = read_pdfs_from_folder(folder)
        print(f"\nJob description: {len(job_description)} characters")
        print(f"Resumes loaded: {len(resumes)}")
        return resumes

def load_job_description(job_description_path: str | Path) -> str:

    try:
        job_description = read_job_description_file(job_description_path)
        print(f"Job description loaded from: {Path(job_description_path).name}")
        print(f"\nJob description: {len(job_description)} characters")
        return job_description
    except (FileNotFoundError, ValueError) as e:
        print(e)
        raise SystemExit(1)

def load_dotenv(
    dotenv_path: Optional[StrPath] = None,
    stream: Optional[IO[str]] = None,
    verbose: bool = False,
    override: bool = False,
    interpolate: bool = True,
    encoding: Optional[str] = "utf-8",
) -> bool:
    """Parse a .env file and then load all the variables found as environment variables.

    Parameters:
        dotenv_path: Absolute or relative path to .env file.
        stream: Text stream (such as `io.StringIO`) with .env content, used if
            `dotenv_path` is `None`.
        verbose: Whether to output a warning the .env file is missing.
        override: Whether to override the system environment variables with the variables
            from the `.env` file.
        encoding: Encoding to be used to read the file.
    Returns:
        Bool: True if at least one environment variable is set else False

    If both `dotenv_path` and `stream` are `None`, `find_dotenv()` is used to find the
    .env file with it's default parameters. If you need to change the default parameters
    of `find_dotenv()`, you can explicitly call `find_dotenv()` and pass the result
    to this function as `dotenv_path`.
    """
    if dotenv_path is None and stream is None:
        dotenv_path = find_dotenv()

    dotenv = DotEnv(
        dotenv_path=dotenv_path,
        stream=stream,
        verbose=verbose,
        interpolate=interpolate,
        override=override,
        encoding=encoding,
    )
    return dotenv.set_as_environment_variables()

subject_instructions = "You can write a subject for a interview process initiation email. \
You are given a message and you need to write a subject for an email that is likely to get a response."

html_instructions = "You can convert a text email body to an HTML email body. \
You are given a text email body which might have some markdown \
and you need to convert it to an HTML email body with simple, clear, compelling layout and design."

subject_writer = Agent(name="Email subject writer", instructions=subject_instructions, model="gpt-4o-mini")
subject_tool = subject_writer.as_tool(tool_name="subject_writer", tool_description="Write a subject for a interview process initiation email")

html_converter = Agent(name="HTML email body converter", instructions=html_instructions, model="gpt-4o-mini")
html_tool = html_converter.as_tool(tool_name="html_converter",tool_description="Convert a text email body to an HTML email body")

instructions ="You are an email formatter and sender. You receive the to_email and body of an email to be sent. \
You first use the subject_writer tool to write a subject for the email, then use the html_converter tool to convert the body to HTML. \
Finally, you use the send_html_email tool to send the email with the subject and HTML body."


@function_tool
def send_html_email(subject: str, html_body: str, to_email: str) -> dict[str, str]:
    """ Send out an email with the given subject and HTML body to all sales prospects """
    print("send html email")
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("dashyabhishek98@gmail.com")  # Change to your verified sender
    to_email = To(to_email)  # Change to your recipient
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}

tools = [subject_tool, html_tool, send_html_email]

emailer_agent = Agent(
    name="Email Manager",
    instructions=instructions,
    tools=tools,
    model="gpt-4o-mini",
    hooks=ConsoleLogHooks(),
    handoff_description="Convert an email to HTML and send it")

# Step 3: Run the Email Manager Agent explicitly!
async def email_to_candidate(candidate_table_data: str, job_description_summary: str):
    # We instruct an LLM block or pass instructions telling Emailer Agent to extract 
    # the exact recipient variables from the Markdown table context generated previously
    print("email_to_candidate")
    outreach_instruction = (
        f"From this candidate table data:\n{candidate_table_data}\n\n"
        f"Extract the 'Email' address value. Draft a warm interview invitation email "
        f"requesting availability for a Technical Screening Round regarding the Job Description: {job_description[:200]}... "
        f"and execute your suite of tools to successfully format and deliver it to that candidate."
    )
    
    with trace("email_to_candidate"):
        email_delivery_result = await Runner.run(emailer_agent, outreach_instruction)
        print("\n--- Outreach Results ---")
        print(email_delivery_result.final_output)




if __name__ == "__main__":

        if not os.environ.get('WANDB_API_KEY'):
            print("WANDB_API_KEY is not set - make sure to export it in your environment or assign it in this script")
            exit(1)
        
        weave.init("abhishek98-psit-kanpur/intro-example")
    
        # 1. Ask the user for inputs at the very beginning
        resumes_folder = input("Enter folder path containing PDF resumes: ").strip().strip('"')
        job_description_path = input("Enter path to job description file (.txt, .md, or .pdf): ").strip().strip('"')

        # Bypassing tool wrappers to parse files using your local machine resources
        job_description = load_job_description(job_description_path)
        resumes = load_resumes(resumes_folder)

        # 3. Check if we actually found any resumes before calling the AI
        if not resumes:
            print("No resumes found to process. Exiting.")
            raise SystemExit(0)

        # Combine the data into a single string message argument
        combined_message = f"Job description:\n{job_description}\n\nResumes:\n{resumes}"
        
        # 5. Execute your async agent workflow
        results = asyncio.run(select_resume(combined_message))

        # 2. Run automated outreach email pipeline using output context data
        candidate_table = results[1].final_output
        asyncio.run(email_to_candidate(candidate_table, job_description))

        # 6. Print the results out
        print("\n--- Agent Results ---")
        for result in results:
            print(result.final_output)
            print()

        


