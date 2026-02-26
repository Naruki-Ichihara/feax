#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "==> Generating API docs..."
npm run api:generate

echo "==> Starting Docusaurus dev server..."
npm run start
