import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Tutorial/documentation sidebar
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      link: {
        type: 'doc',
        id: 'getting-started/index',
      },
      items: [
        'getting-started/overview',
        'getting-started/installation',
      ],
    },
    {
      type: 'category',
      label: 'Basic Tutorials',
      link: {
        type: 'doc',
        id: 'basic/index',
      },
      items: [
        'basic/linear_elasticity',
        'basic/jit_transform',
        'basic/vmap_transform',
        'basic/hyperelasticity',
      ],
    },
    {
      type: 'category',
      label: 'Advanced Tutorials',
      link: {
        type: 'doc',
        id: 'advanced/index',
      },
      items: [
        'advanced/periodic_boundary_conditions',
        'advanced/lattice_homogenization',
        'advanced/adaptive_topology_optimization',
        'advanced/cohesive_fracture',
      ],
    },
  ],

  // API Reference sidebar - manually configured
  apiSidebar: [
    'api/index',
    'api/reference/feax/assembler',
    'api/reference/feax/basis',
    'api/reference/feax/DCboundary',
    'api/reference/feax/fe',
    'api/reference/feax/internal_vars',
    'api/reference/feax/mesh',
    'api/reference/feax/problem',
    'api/reference/feax/solver',
    'api/reference/feax/utils',
    {
      type: 'category',
      label: 'Solvers',
      link: {
        type: 'doc',
        id: 'api/reference/feax/solvers/index',
      },
      items: [
        'api/reference/feax/solvers/options',
        'api/reference/feax/solvers/linear',
        'api/reference/feax/solvers/newton',
        'api/reference/feax/solvers/matrix_free',
        'api/reference/feax/solvers/common',
        'api/reference/feax/solvers/reduced',
      ],
    },
    {
      type: 'category',
      label: 'Topology Optimization (gene)',
      link: {
        type: 'doc',
        id: 'api/reference/feax/gene/index',
      },
      items: [
        'api/reference/feax/gene/optimizer',
        'api/reference/feax/gene/adaptive',
        'api/reference/feax/gene/filters',
        'api/reference/feax/gene/responses',
        'api/reference/feax/gene/mdmm',
      ],
    },
    {
      type: 'category',
      label: 'Mechanics',
      link: {
        type: 'doc',
        id: 'api/reference/feax/mechanics/index',
      },
      items: [
        'api/reference/feax/mechanics/cohesive',
      ],
    },
    {
      type: 'category',
      label: 'Flat Toolkit',
      link: {
        type: 'doc',
        id: 'api/reference/feax/flat/index',
      },
      items: [
        'api/reference/feax/flat/graph',
        'api/reference/feax/flat/pbc',
        'api/reference/feax/flat/solver',
        'api/reference/feax/flat/unitcell',
        'api/reference/feax/flat/utils',
      ],
    },
  ],
};

export default sidebars;
