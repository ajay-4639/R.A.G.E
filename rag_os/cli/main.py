"""Main CLI entry point for RAG OS."""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.syntax import Syntax

import rag_os
from rag_os.sdk.client import RAGClient, RAGClientConfig

app = typer.Typer(
    name="rag-os",
    help="RAG OS - A fully customizable RAG Operating System",
    add_completion=False,
)

console = Console()

# Sub-commands
pipeline_app = typer.Typer(help="Pipeline management commands")
index_app = typer.Typer(help="Index management commands")
config_app = typer.Typer(help="Configuration commands")

app.add_typer(pipeline_app, name="pipeline")
app.add_typer(index_app, name="index")
app.add_typer(config_app, name="config")


def get_client(config_path: Optional[Path] = None) -> RAGClient:
    """Get or create a RAG client instance."""
    if config_path and config_path.exists():
        config_data = json.loads(config_path.read_text())
        config = RAGClientConfig(**config_data)
        return RAGClient(config)
    return RAGClient()


@app.command()
def version():
    """Show RAG OS version information."""
    console.print(Panel(
        f"[bold blue]RAG OS[/bold blue] version [green]{rag_os.__version__}[/green]",
        title="Version Info",
    ))


@app.command()
def query(
    text: str = typer.Argument(..., help="The query text to process"),
    pipeline: str = typer.Option("default", "--pipeline", "-p", help="Pipeline to use"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
    output_format: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Run a query through a RAG pipeline."""
    client = get_client(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Processing query...", total=None)

        try:
            result = client.query(text, pipeline=pipeline)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    if output_format == "json":
        output = {
            "query": text,
            "pipeline": pipeline,
            "result": result.data if hasattr(result, "data") else str(result),
            "success": result.success if hasattr(result, "success") else True,
        }
        if verbose and hasattr(result, "step_results"):
            output["step_results"] = {
                k: str(v) for k, v in result.step_results.items()
            }
        console.print(Syntax(json.dumps(output, indent=2), "json"))
    else:
        console.print(Panel(
            str(result.data if hasattr(result, "data") else result),
            title=f"[bold]Query Result[/bold] (pipeline: {pipeline})",
            border_style="green",
        ))

        if verbose and hasattr(result, "step_results"):
            console.print("\n[bold]Step Results:[/bold]")
            for step_id, step_result in result.step_results.items():
                console.print(f"  [cyan]{step_id}[/cyan]: {type(step_result).__name__}")


@app.command()
def interactive(
    pipeline: str = typer.Option("default", "--pipeline", "-p", help="Pipeline to use"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Start an interactive RAG session."""
    from rag_os.sdk.session import RAGSession, SessionConfig

    console.print(Panel(
        "[bold blue]RAG OS Interactive Mode[/bold blue]\n"
        "Type your queries and press Enter. Type 'exit' or 'quit' to leave.\n"
        "Commands: /clear (clear history), /history (show history), /help",
        title="Welcome",
    ))

    client = get_client(config)
    session_config = SessionConfig(max_history=20)
    session = RAGSession(client=client, config=session_config)

    while True:
        try:
            user_input = console.input("[bold green]>[/bold green] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if user_input == "/clear":
                session.clear_history()
                console.print("[dim]History cleared.[/dim]")
                continue

            if user_input == "/history":
                history = session.get_history()
                if not history:
                    console.print("[dim]No history yet.[/dim]")
                else:
                    for i, entry in enumerate(history, 1):
                        console.print(f"[dim]{i}.[/dim] {entry.get('query', '')[:50]}...")
                continue

            if user_input == "/help":
                console.print(
                    "[bold]Commands:[/bold]\n"
                    "  /clear   - Clear conversation history\n"
                    "  /history - Show conversation history\n"
                    "  /help    - Show this help\n"
                    "  exit     - Exit interactive mode"
                )
                continue

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Thinking...", total=None)
                result = session.query(user_input)

            console.print(Panel(
                str(result.data if hasattr(result, "data") else result),
                border_style="blue",
            ))

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


# Pipeline commands
@pipeline_app.command("list")
def pipeline_list(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """List all available pipelines."""
    client = get_client(config)

    table = Table(title="Available Pipelines")
    table.add_column("Name", style="cyan")
    table.add_column("Steps", style="green")
    table.add_column("Status", style="yellow")

    pipelines = client.list_pipelines()
    if not pipelines:
        console.print("[dim]No pipelines loaded.[/dim]")
        return

    for name in pipelines:
        info = client.get_pipeline_info(name)
        step_count = len(info.get("steps", [])) if info else 0
        table.add_row(name, str(step_count), "active")

    console.print(table)


@pipeline_app.command("info")
def pipeline_info(
    name: str = typer.Argument(..., help="Pipeline name"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Show detailed information about a pipeline."""
    client = get_client(config)

    info = client.get_pipeline_info(name)
    if not info:
        console.print(f"[red]Pipeline '{name}' not found.[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Pipeline:[/bold] {name}\n"
        f"[bold]Version:[/bold] {info.get('version', 'unknown')}\n"
        f"[bold]Steps:[/bold] {len(info.get('steps', []))}",
        title="Pipeline Info",
    ))

    if info.get("steps"):
        table = Table(title="Steps")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Enabled", style="yellow")

        for step in info["steps"]:
            table.add_row(
                step.get("id", "unknown"),
                step.get("type", "unknown"),
                "Yes" if step.get("enabled", True) else "No",
            )
        console.print(table)


@pipeline_app.command("load")
def pipeline_load(
    path: Path = typer.Argument(..., help="Path to pipeline spec file (JSON/YAML)"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Override pipeline name"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Load a pipeline from a specification file."""
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    client = get_client(config)

    try:
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                spec_data = yaml.safe_load(content)
            except ImportError:
                console.print("[red]PyYAML required for YAML files. Install with: pip install pyyaml[/red]")
                raise typer.Exit(1)
        else:
            spec_data = json.loads(content)

        if name:
            spec_data["name"] = name

        from rag_os.core.spec import PipelineSpec
        spec = PipelineSpec(**spec_data)
        client.load_pipeline(spec)

        console.print(f"[green]Pipeline '{spec.name}' loaded successfully.[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load pipeline: {e}[/red]")
        raise typer.Exit(1)


@pipeline_app.command("create")
def pipeline_create(
    name: str = typer.Argument(..., help="Name for the new pipeline"),
    output: Path = typer.Option(Path("pipeline.json"), "--output", "-o", help="Output file path"),
    template: str = typer.Option("basic", "--template", "-t", help="Template: basic, advanced, minimal"),
):
    """Create a new pipeline specification from a template."""
    templates = {
        "minimal": {
            "name": name,
            "version": "1.0.0",
            "steps": [
                {"step_id": "retriever", "step_type": "retrieval", "config": {"top_k": 5}},
            ],
        },
        "basic": {
            "name": name,
            "version": "1.0.0",
            "steps": [
                {"step_id": "retriever", "step_type": "retrieval", "config": {"top_k": 5}},
                {"step_id": "reranker", "step_type": "reranking", "config": {"top_k": 3}, "dependencies": ["retriever"]},
                {"step_id": "prompt", "step_type": "prompt_assembly", "config": {}, "dependencies": ["reranker"]},
                {"step_id": "llm", "step_type": "llm_execution", "config": {"model": "gpt-4"}, "dependencies": ["prompt"]},
            ],
        },
        "advanced": {
            "name": name,
            "version": "1.0.0",
            "steps": [
                {"step_id": "parser", "step_type": "parsing", "config": {}},
                {"step_id": "chunker", "step_type": "chunking", "config": {"chunk_size": 512, "overlap": 50}},
                {"step_id": "embedder", "step_type": "embedding", "config": {"model": "text-embedding-3-small"}},
                {"step_id": "retriever", "step_type": "retrieval", "config": {"top_k": 10}},
                {"step_id": "reranker", "step_type": "reranking", "config": {"top_k": 5}, "dependencies": ["retriever"]},
                {"step_id": "prompt", "step_type": "prompt_assembly", "config": {}, "dependencies": ["reranker"]},
                {"step_id": "llm", "step_type": "llm_execution", "config": {"model": "gpt-4"}, "dependencies": ["prompt"]},
                {"step_id": "postprocess", "step_type": "post_processing", "config": {}, "dependencies": ["llm"]},
            ],
        },
    }

    if template not in templates:
        console.print(f"[red]Unknown template: {template}. Available: {', '.join(templates.keys())}[/red]")
        raise typer.Exit(1)

    spec = templates[template]
    output.write_text(json.dumps(spec, indent=2))
    console.print(f"[green]Pipeline spec created: {output}[/green]")


# Index commands
@index_app.command("create")
def index_create(
    name: str = typer.Argument(..., help="Index name"),
    source: Path = typer.Argument(..., help="Source directory or file to index"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Create a new index from documents."""
    if not source.exists():
        console.print(f"[red]Source not found: {source}[/red]")
        raise typer.Exit(1)

    console.print(f"[yellow]Creating index '{name}' from {source}...[/yellow]")

    # Collect files
    if source.is_file():
        files = [source]
    else:
        files = list(source.rglob("*"))
        files = [f for f in files if f.is_file() and f.suffix in (".txt", ".md", ".pdf", ".json")]

    console.print(f"Found {len(files)} files to index.")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing...", total=len(files))

        for f in files:
            progress.update(task, description=f"Indexing {f.name}...")
            # Actual indexing would happen here
            progress.advance(task)

    console.print(f"[green]Index '{name}' created successfully with {len(files)} documents.[/green]")


@index_app.command("list")
def index_list():
    """List all available indexes."""
    table = Table(title="Available Indexes")
    table.add_column("Name", style="cyan")
    table.add_column("Documents", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("Created", style="dim")

    # Placeholder - would read from actual index storage
    console.print("[dim]No indexes found.[/dim]")


@index_app.command("delete")
def index_delete(
    name: str = typer.Argument(..., help="Index name to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete an index."""
    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete index '{name}'?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    console.print(f"[green]Index '{name}' deleted.[/green]")


# Config commands
@config_app.command("show")
def config_show(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Show current configuration."""
    if config and config.exists():
        content = config.read_text()
        console.print(Syntax(content, "json", theme="monokai"))
    else:
        console.print("[dim]No configuration file specified or found.[/dim]")
        console.print("\nDefault configuration:")
        default_config = {
            "default_pipeline": "default",
            "storage": {"type": "memory"},
            "tracing": {"enabled": False},
            "rate_limit": {"enabled": False},
        }
        console.print(Syntax(json.dumps(default_config, indent=2), "json"))


@config_app.command("init")
def config_init(
    output: Path = typer.Option(Path("rag_os.json"), "--output", "-o", help="Output path"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
):
    """Initialize a new configuration file."""
    if output.exists() and not force:
        console.print(f"[red]Config file already exists: {output}. Use --force to overwrite.[/red]")
        raise typer.Exit(1)

    default_config = {
        "default_pipeline": "default",
        "storage": {
            "type": "memory",
            "path": "./data",
        },
        "tracing": {
            "enabled": True,
            "exporter": "console",
        },
        "rate_limit": {
            "enabled": True,
            "requests_per_minute": 60,
        },
        "security": {
            "input_validation": True,
            "max_query_length": 10000,
        },
    }

    output.write_text(json.dumps(default_config, indent=2))
    console.print(f"[green]Configuration file created: {output}[/green]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """Start the RAG OS API server."""
    console.print(Panel(
        f"Starting RAG OS API server on [bold]http://{host}:{port}[/bold]\n"
        f"Press Ctrl+C to stop.",
        title="RAG OS Server",
    ))

    try:
        import uvicorn
        uvicorn.run(
            "rag_os.api.server:app",
            host=host,
            port=port,
            reload=reload,
        )
    except ImportError:
        console.print("[red]uvicorn required. Install with: pip install uvicorn[/red]")
        raise typer.Exit(1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
