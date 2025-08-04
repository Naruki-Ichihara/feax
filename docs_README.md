# FEAX Documentation

This directory contains the automatically generated API documentation for FEAX.

## Generating Documentation

### Local Development

To generate the documentation locally:

```bash
# Generate documentation
python generate_docs.py

# Serve documentation locally (opens browser automatically)
python docs_serve.py
```

The documentation will be available at `http://localhost:8000`.

### Automated Deployment

Documentation is automatically built and deployed to GitHub Pages on every push to the main branch. The workflow is defined in `.github/workflows/docs.yml`.

## Configuration

- **`pdoc.yml`**: Configuration file for pdoc documentation generation
- **`generate_docs.py`**: Script for local documentation generation
- **`docs_serve.py`**: Simple HTTP server for viewing docs locally

## Features

The generated documentation includes:

- **Full API Reference**: All public classes, functions, and methods
- **Mathematical Formulas**: LaTeX math rendering with MathJax
- **Search Functionality**: Built-in search across all documentation
- **Responsive Design**: Works on desktop and mobile devices
- **Cross-references**: Automatic linking between related components

## Viewing Online

The latest documentation is available at: `https://[your-username].github.io/[repository-name]/`

Replace `[your-username]` and `[repository-name]` with your actual GitHub username and repository name.

## Customization

The documentation appearance can be customized by:

1. Modifying `pdoc.yml` configuration
2. Adding custom CSS in the generation scripts
3. Using custom templates (see pdoc documentation)

## Troubleshooting

If documentation generation fails:

1. Ensure all dependencies are installed: `pip install -e .`
2. Check that the `feax` module can be imported: `python -c "import feax"`
3. Verify pdoc is installed: `pdoc --version`

For GitHub Pages deployment issues, check the Actions tab in your repository.