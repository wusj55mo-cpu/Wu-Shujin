from __future__ import annotations

import json
from typing import Any, Callable

from backend.artifacts.filesystem import ArtifactRun
from backend.pipelines.authorship import add_author_credit
from backend.pipelines.common import (
    PipelineError,
    PipelineResult,
    extract_fenced_or_raw,
    format_citations,
    format_log,
)
from backend.pipelines.html_to_pdf import (
    A4_PAGE,
    ConvertError,
    HtmlToPdfConverter,
)
from backend.providers.base import (
    ModelProviderError,
    TextGenerationProvider,
    TextGenerationRequest,
)
from backend.timing import RunTimingRecorder, measure_stage


def run_essay_html_pipeline(
    *,
    artifact_run: ArtifactRun,
    model_profile: dict[str, Any],
    model_provider: TextGenerationProvider,
    pdf_converter: HtmlToPdfConverter,
    task_text: str,
    context_bundle: str,
    output_preference: str | None,
    options: dict[str, Any] | None,
    search: dict[str, Any],
    max_output_tokens: int,
    emit_event: Callable[[str, str], None] | None = None,
    timing: RunTimingRecorder | None = None,
) -> PipelineResult:
    log_lines = [
        "Pipeline: essay_html",
        f"Output preference: {output_preference or 'pdf'}",
        "Stage: generate_source",
    ]
    system_prompt, user_prompt = build_essay_html_prompt(
        task_text=task_text,
        context_bundle=context_bundle,
        options=options or {},
        search=search,
    )
    if emit_event:
        emit_event("generate_source", "Generating essay HTML source")

    try:
        with measure_stage(timing, "provider_generation"):
            raw_output = model_provider.generate_text(
                TextGenerationRequest(
                    profile=model_profile,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=0.2,
                )
            )
    except ModelProviderError as exc:
        raise PipelineError(
            exc.code,
            exc.message,
            stage="generate_source",
            log_lines=log_lines,
        ) from exc
    except Exception as exc:
        raise PipelineError(
            "provider_unavailable",
            "The model provider request failed.",
            stage="generate_source",
            log_lines=log_lines,
        ) from exc

    source = extract_fenced_or_raw(raw_output, accepted_languages={"html", "htm"})
    source = add_author_credit(source.strip()) + "\n"
    html_path = artifact_run.write_output(
        "main.html",
        source,
        kind="source",
        media_type="text/html",
    )
    log_lines.extend(["Output: output/main.html", "Stage: convert_pdf"])
    if emit_event:
        emit_event("convert_pdf", "Converting essay HTML to PDF")

    pdf_path = artifact_run.run_dir / "output" / "main.pdf"
    try:
        with measure_stage(timing, "convert_pdf"):
            convert_result = pdf_converter.convert(
                html_path=html_path,
                pdf_path=pdf_path,
                page_config=A4_PAGE,
            )
    except ConvertError as exc:
        artifact_run.write_log("convert.log", exc.log_text)
        raise PipelineError(
            "convert_failed",
            exc.message,
            stage="convert_pdf",
            log_lines=log_lines,
        ) from exc
    except Exception as exc:
        artifact_run.write_log(
            "convert.log",
            "HTML-to-PDF converter raised an unexpected error.\n",
        )
        raise PipelineError(
            "convert_failed",
            "HTML-to-PDF conversion failed.",
            stage="convert_pdf",
            log_lines=log_lines,
        ) from exc

    artifact_run.write_log("convert.log", convert_result.log_text)
    artifact_run.write_output(
        "main.pdf",
        convert_result.pdf_path.read_bytes(),
        kind="pdf",
        media_type="application/pdf",
    )
    log_lines.extend(["Convert: pdf_ok", "Output: output/main.pdf"])
    return PipelineResult(log_text=format_log(log_lines))


def build_essay_html_prompt(
    *,
    task_text: str,
    context_bundle: str,
    options: dict[str, Any],
    search: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = (
        "You are an expert teaching assistant writing complete academic essay and "
        "report deliverables as self-contained HTML documents. Return only the "
        "requested HTML source, with no surrounding explanation."
    )
    output_instruction = """        
    You are writing a formal academic assignment report for BA3073 From AI Agent to Cloud Service. Maximum report length: 20 pages.
Structure strictly follows the outline below, write with sufficient technical detail and include the author's hands-on debugging thinking process, rather than only listing facts.
1. Introduction & Learning Objectives
Introduce the AI Learning Assistant project, its core function as an AI agent service. Clearly explain the progressive learning aims of Task1, Task2 and Task3: starting from local containerised development, moving to cloud public deployment, then extending to HTTPS reverse proxy and external access control.

Task1: Local Docker deployment on Windows.
The backend service runs on localhost:14242, access URL http://127.0.0.1:14242/ui/. The whole service is packaged and run inside Docker containers. After starting the stack, curl functional test was executed and returned HTTP 200 response. Test records were saved in local-ui-check.txt with final test status PASS. Describe the local development workflow: container environment setup, service dependency management and local interface verification.

Task2: Aliyun ECS public IP deployment.
The AI assistant is deployed on an Aliyun ECS instance with public IP 8.155.136.210. Port 14242 is configured in the security group, and access is restricted to the author's own client public IP only for security compliance. Access URL: http://8.155.136.210:14242/ui/. The ecs-docker-check script was executed and ecs-docker-check.txt evidence was saved. Explain the migration process from local Docker environment to remote cloud server, environment synchronisation and remote service health verification.

Task3: Custom domain, HTTPS reverse proxy and HTTP Basic Authentication.
The author registered and configured domain wsj26.xyz, created DNS A record pointing to the ECS public IP. Nginx was installed as the HTTPS reverse proxy, and Certbot was used to issue trusted SSL certificates for TLS encryption.
An independent HTTP Basic Auth layer was configured inside Nginx, this authentication mechanism is fully separated from the BNBU application login system, with test user wsjuser created using htpasswd tool.
Curl verification evidence:
- Without credential: returns HTTP/1.1 401 Unauthorized with WWW-Authenticate header.
- With valid credential: returns HTTP/1.1 200 OK.

Describe the challenges encountered during this domain deployment, together with the troubleshooting logic attempted:
Challenge A: Repeated Basic Auth pop-up after logging into the BNBU UI. Basic Auth is a stateless authentication scheme. After the user enters credentials and enters the frontend UI, every asynchronous /api request initiated by the SPA frontend will pass through Nginx and trigger authentication validation repeatedly. Multiple mitigation ideas were considered, including forwarding the Authorization header and exempting the /api path from auth checks, but both approaches would break the integrity of reverse-proxy access control, so neither solution was adopted.
Challenge B: Browser and device inconsistency. Even though the server-side curl test confirmed the Nginx authentication logic worked correctly, desktop Chrome browsers would sometimes cache authentication credentials and skip the pop-up prompt automatically. Mobile browsers showed inconsistent behaviour; many mobile clients could not render the native Basic Auth prompt properly, leading to blank pages or connection failures. The website was occasionally unreachable on mobile networks.
Challenge C: SSL handshake warning: SSL_do_handshake() failed (bad key share) on older mobile clients. This error was identified as a TLS version and key exchange compatibility issue between legacy mobile TLS stacks and modern Certbot-generated configuration. Multiple client devices were tested to confirm this was a client-side compatibility limitation rather than certificate misconfiguration.

Then add:
System workflow explanation: Explain end-to-end how the whole stack works. Describe the role of Docker container isolation, FastAPI backend service, SPA frontend, Nginx reverse proxy, HTTPS TLS encryption and HTTP Basic Auth access control, and the data flow between each component.

Improvement section: Discuss possible future improvements for this deployment. Include feasible technical optimisations, such as session-aware access control, TLS compatibility tuning, and separation of authentication logic.

Important rules:
Do NOT add extra sections such as cloud cost analysis, pricing, billing or other unrequired topics.
Do NOT add any placeholders for screenshots, image insertion notes or image markers anywhere in the report.
Report tone: formal academic report, third person, clear technical explanation, no excessive markdown.
Do NOT invent fake evidence. All discussion must match the deployment facts above.

    """
    user_prompt = "\n\n".join(
        [
            "[Assignment Task]\n" + task_text.strip(),
            "[Prepared Context]\n" + context_bundle.strip(),
            "[Search Citations]\n" + format_citations(search),
            "[Options]\n" + json.dumps(options, sort_keys=True),
            "[Output Contract]\n" + output_instruction,
        ]
    )
    return system_prompt, user_prompt
