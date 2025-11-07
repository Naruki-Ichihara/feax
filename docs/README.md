# FEAX Documentation

This directory contains the Docusaurus documentation site for FEAX (Finite Element Analysis with JAX).

Built with [Docusaurus 3.9.2](https://docusaurus.io/).

## Installation

```bash
npm install
```

## Local Development

```bash
npm start
```

This command starts a local development server at `http://localhost:3000/feax/`. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch via GitHub Actions (`.github/workflows/deploy-docs.yml`).

Live site: https://naruki-ichihara.github.io/feax/

## Structure

- `docs/` - Documentation markdown files
- `blog/` - Blog posts
- `src/` - React components and custom pages
- `static/` - Static assets (images, files, etc.)
- `docusaurus.config.ts` - Docusaurus configuration
- `sidebars.ts` - Sidebar navigation configuration
