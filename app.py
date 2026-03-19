import os
import tempfile
from typing import Any, Callable, List, Optional, Tuple

import gradio as gr

from rag_app import build_rag_chain, build_pdf_web_chain


def load_api_key() -> str:
    """
    Load GOOGLE_API_KEY from .streamlit/secrets.toml.
    Raises a Gradio-friendly error if missing.
    """
    secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")

    if not os.path.exists(secrets_path):
        raise gr.Error("Could not find .streamlit/secrets.toml with GOOGLE_API_KEY.")

    api_key: Optional[str] = None

    with open(secrets_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("GOOGLE_API_KEY"):
                # Expect format: GOOGLE_API_KEY="value"
                _, value = stripped.split("=", 1)
                value = value.strip().strip('"').strip("'")
                api_key = value
                break

    if not api_key:
        raise gr.Error("GOOGLE_API_KEY not set in .streamlit/secrets.toml.")

    return api_key


def _build_chain(
    files: List[Any],
    mode: str,
) -> Tuple[Callable[[str], dict], List[str]]:
    """
    Build the appropriate RAG chain from uploaded PDF files.
    Returns the chain and a list of temporary PDF paths.
    """
    if not files:
        raise gr.Error("Please upload at least one PDF.")

    api_key = load_api_key()

    pdf_paths: List[str] = []

    for f in files:
        # Gradio returns file objects that typically include a local path.
        # Support the common shapes: FileData-like objects, dicts, or plain paths.
        src_path = None
        if isinstance(f, str):
            src_path = f
        elif isinstance(f, dict) and "path" in f:
            src_path = f["path"]
        else:
            src_path = getattr(f, "path", None) or getattr(f, "name", None)

        if not src_path or not os.path.exists(src_path):
            raise gr.Error("One of the uploaded PDFs could not be read. Please re-upload the file(s).")

        # Persist uploaded PDFs to temporary files for the loaders
        with open(src_path, "rb") as src, tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(src.read())
            pdf_paths.append(tmp.name)

    if mode == "PDF Only":
        chain = build_rag_chain(pdf_paths, api_key)
    else:
        chain = build_pdf_web_chain(pdf_paths, api_key)

    return chain, pdf_paths


def chat(
    message: str,
    history: List[dict],
    files: List[gr.File],
    mode: str,
    chain: Optional[Callable],
    pdf_paths: Optional[List[str]],
):
    """
    Main chat handler. Builds the chain on first use and then reuses it.
    Keeps full history of previous questions and answers in the UI.
    """
    if not message:
        return history, chain, pdf_paths

    # Build chain lazily the first time user asks a question
    if chain is None or not pdf_paths:
        chain, pdf_paths = _build_chain(files, mode)

    try:
        result = chain(message)
        answer = result["answer"]
    except Exception as e:
        # Show a friendly error in the chat instead of crashing the UI
        answer = (
            "There was an error calling the Google API.\n\n"
            f"Details: {e}"
        )

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]

    return history, chain, pdf_paths


css_string = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gr-button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border: none;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .gr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .gr-textbox {
        border-radius: 15px;
        border: 2px solid #ddd;
        transition: border-color 0.3s ease;
    }
    .gr-textbox:focus {
        border-color: #4ECDC4;
        box-shadow: 0 0 10px rgba(78, 205, 196, 0.3);
    }
    .gr-file {
        border-radius: 15px;
        border: 2px dashed #4ECDC4;
        background: rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .gr-file:hover {
        border-color: #FF6B6B;
        background: rgba(255, 107, 107, 0.1);
    }
    .gr-radio {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 10px;
    }
    .gr-chatbot {
        border-radius: 15px;
        border: 2px solid #ddd;
        background: rgba(255, 255, 255, 0.9);
    }
    .gr-markdown {
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-size: 1.5em;
        margin-bottom: 20px;
    }
    .gr-row {
        margin-bottom: 20px;
    }
    """

with gr.Blocks(title="Ask Your PDF") as demo:
    gr.Markdown(
        "## Ask Your PDF\n\n"

        "Upload one or more PDFs, then chat with your documents.\n\n"
    )

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(
                label="Upload PDF(s)",
                file_types=[".pdf"],
                file_count="multiple",
            )

            mode = gr.Radio(
                choices=["PDF Only", "PDF + Web Search"],
                value="PDF Only",
                label="Mode",
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat with your PDFs")
            msg = gr.Textbox(
                label="Ask a question about the PDF...",
                placeholder="Type your question and press Enter",
            )
            clear_btn = gr.Button("Clear conversation")

    # Per-user state: the built chain and the list of temp PDF paths
    state_chain = gr.State(None)
    state_pdf_paths = gr.State([])

    msg.submit(
        fn=chat,
        inputs=[msg, chatbot, files, mode, state_chain, state_pdf_paths],
        outputs=[chatbot, state_chain, state_pdf_paths],
    )

    clear_btn.click(lambda: ([], None, []), None, [chatbot, state_chain, state_pdf_paths])

    # Clear the message box after submit
    msg.submit(lambda: "", None, msg)


if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft(), css=css_string)
