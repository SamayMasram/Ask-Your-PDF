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
    /* Dark theme as default */
    .gradio-container {
        background: #0f172a;
        color: #e2e8f0;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gradio-container.light-mode {
        background: #f6f8fb;
        color: #1f2937;
    }
    .gradio-container.dark-mode {
        background: #0f172a;
        color: #e2e8f0;
    }
    .gradio-container.dark-mode .gr-markdown,
    .gradio-container.dark-mode .gradio-container .section-title {
        color: #e2e8f0;
    }
    .gradio-container.light-mode .gr-markdown,
    .gradio-container.light-mode .gradio-container .section-title {
        color: #1f2937;
    }
    .gr-block {
        padding: 24px;
        margin: 0 auto;
        max-width: 1100px;
    }
    .gradio-container .section-title,
    .gr-markdown {
        color: #1f2937;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 16px;
    }
    .gradio-container .description {
        color: #4b5563;
        font-size: 1rem;
        margin-bottom: 24px;
    }
    .gr-button {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        border: none;
        color: #fff;
        border-radius: 10px;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 10px 20px;
        box-shadow: 0 8px 20px rgba(31, 41, 55, 0.12);
        transition: all 0.2s ease;
    }
    .gr-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(31, 41, 55, 0.18);
    }
    .gr-textbox, .gr-file, .gr-radio, .gr-chatbot {
        border-radius: 12px !important;
        border: 1px solid #d1d5db !important;
        background: #fff !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }
    .gradio-container.dark-mode .gr-textbox,
    .gradio-container.dark-mode .gr-file,
    .gradio-container.dark-mode .gr-radio,
    .gradio-container.dark-mode .gr-chatbot {
        border-color: #334155 !important;
        background: #1e293b !important;
        color: #e2e8f0 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .gr-textbox:focus, .gr-file:focus, .gr-radio:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18) !important;
    }
    .gr-chatbot {
        padding: 16px;
        min-height: 360px;
        background: #ffffff !important;
    }
    .gr-row {
        gap: 20px;
        margin-bottom: 18px;
    }
    .gradio-container .footer {
        color: #6b7280;
        font-size: 0.88rem;
        text-align: center;
        margin-top: 18px;
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
                choices=["PDF Only", "Web Search"],
                value="PDF Only",
                label="Mode",
            )

            theme = gr.Radio(
                choices=["Light", "Dark"],
                value="Dark",
                label="Theme",
            )

            theme_style = gr.HTML(visible=False)

            def apply_theme(selected_theme: str) -> str:
                selected = selected_theme.lower()
                return f"""
                <script>
                const root = document.querySelector('.gradio-container');
                if (root) {{
                    root.classList.remove('light-mode', 'dark-mode');
                    root.classList.add('{selected}-mode');
                }}
                </script>
                """

            theme.change(fn=apply_theme, inputs=[theme], outputs=[theme_style])


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
    