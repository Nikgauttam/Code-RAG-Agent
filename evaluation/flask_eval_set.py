"""Flask eval set — 30 hand-labeled questions for the pallets/flask codebase.

Run with:
    python -m evaluation.run_eval --repo /tmp/flask_eval --eval flask

Each entry: (question, [expected_relative_paths]).
A retrieval hit = at least one expected file appears in top-k results.
"""

FLASK_EVAL_SET: list[tuple[str, list[str]]] = [
    # --- sessions ---
    (
        "How does Flask implement secure cookie sessions?",
        ["src/flask/sessions.py"],
    ),
    (
        "Where is the session interface defined?",
        ["src/flask/sessions.py"],
    ),
    (
        "How are sessions serialized and deserialized from cookies?",
        ["src/flask/sessions.py"],
    ),
    # --- request context ---
    (
        "How does Flask push and pop the request context?",
        ["src/flask/ctx.py"],
    ),
    (
        "Where is the application context implemented?",
        ["src/flask/ctx.py"],
    ),
    (
        "How does after_this_request work?",
        ["src/flask/ctx.py"],
    ),
    (
        "Where is has_request_context defined?",
        ["src/flask/ctx.py"],
    ),
    # --- routing / app ---
    (
        "Where is the main Flask application class defined?",
        ["src/flask/app.py", "src/flask/sansio/app.py"],
    ),
    (
        "How does Flask register URL routes?",
        ["src/flask/app.py", "src/flask/sansio/scaffold.py"],
    ),
    (
        "How are blueprints registered on the Flask app?",
        ["src/flask/blueprints.py", "src/flask/sansio/blueprints.py"],
    ),
    # --- templating ---
    (
        "How does Flask render Jinja2 templates?",
        ["src/flask/templating.py"],
    ),
    (
        "Where is render_template defined?",
        ["src/flask/templating.py"],
    ),
    (
        "How does Flask stream template responses?",
        ["src/flask/templating.py"],
    ),
    # --- config ---
    (
        "How does Flask load configuration from a file?",
        ["src/flask/config.py"],
    ),
    (
        "Where is the Config class defined?",
        ["src/flask/config.py"],
    ),
    # --- helpers ---
    (
        "How does url_for generate URLs for endpoints?",
        ["src/flask/helpers.py"],
    ),
    (
        "How does Flask send static files?",
        ["src/flask/helpers.py"],
    ),
    (
        "Where is flash and get_flashed_messages implemented?",
        ["src/flask/helpers.py"],
    ),
    (
        "How does stream_with_context work?",
        ["src/flask/helpers.py"],
    ),
    (
        "Where is the redirect helper defined?",
        ["src/flask/helpers.py"],
    ),
    # --- CLI ---
    (
        "How does the Flask CLI locate the app?",
        ["src/flask/cli.py"],
    ),
    (
        "Where is the run command implemented?",
        ["src/flask/cli.py"],
    ),
    (
        "How does Flask load .env files?",
        ["src/flask/cli.py"],
    ),
    # --- testing ---
    (
        "How does FlaskClient simulate HTTP requests in tests?",
        ["src/flask/testing.py"],
    ),
    (
        "Where is the test client defined?",
        ["src/flask/testing.py"],
    ),
    # --- views ---
    (
        "How does Flask implement class-based views?",
        ["src/flask/views.py"],
    ),
    (
        "Where is MethodView defined?",
        ["src/flask/views.py"],
    ),
    # --- signals ---
    (
        "How does Flask use signals for request lifecycle events?",
        ["src/flask/signals.py"],
    ),
    # --- logging ---
    (
        "How does Flask configure its default logger?",
        ["src/flask/logging.py"],
    ),
    # --- globals ---
    (
        "Where are the request and g globals defined?",
        ["src/flask/globals.py"],
    ),
]
